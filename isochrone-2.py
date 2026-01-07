import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.graph import MCP_Geometric
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_erosion, binary_dilation
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import os

#=========================================================================
class USER_CONFIG:
    #檔案設定
    FILE_EXISTING = "site_existing.jpg"   #設計前圖片
    FILE_PROPOSED = "site_design.jpg"     #設計後圖片
    
    #起點座標 (Y, X) - Y=垂直, X=水平
    START_POINT = (500, 1100) 

    #顏色定義 (RGB)
    COLOR_PALETTE = {
        'WALL':   [8, 8, 10],      #牆壁 
        'ROAD':   [255, 255, 255], #馬路
        'PLAZA':  [100, 211, 73],  #綠地
        'PORTAL': [226, 1, 19]     #斑馬線
    }
    COLOR_TOLERANCE = 40.0 #顏色容許誤差值

    #分析參數
    PIXELS_PER_MINUTE = 200.0       #1分鐘走多少像素
    METERS_PER_PIXEL = 0.8          #1像素距離多少公尺
    MAX_VISUAL_TIME = 15.0          #熱力圖最大顯示分鐘數(與下同)
    LEVELS = [1, 3, 5, 8, 10, 15]   #顯示熱力圖等時圈分鐘數
    PENALTY_BASE = 0.5         
    PENALTY_FACTOR = 0.06      

#=========================================================================

def get_color_mask(img_rgb, target_color, tolerance):               #計算顏色遮罩
    target = np.array(target_color)
    dist = np.linalg.norm(img_rgb - target, axis=2)
    return dist < tolerance

def run_analysis_pipeline(image_path, output_folder, run_name):     #執行單張圖的完整分析流程
    cfg = USER_CONFIG
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"\n[{run_name}] 正在處理: {image_path}")
    
    try:
        img_rgb = io.imread(image_path)
        if img_rgb.shape[-1] == 4: img_rgb = img_rgb[..., :3]
    except FileNotFoundError:
        print(f" 錯誤: 找不到檔案。 {image_path}")
        return None

    print(f"[{run_name}] 進行顏色與雜訊處理...")
    mask_wall   = get_color_mask(img_rgb, cfg.COLOR_PALETTE['WALL'],   cfg.COLOR_TOLERANCE) #用顏色分類出不同元素
    mask_road   = get_color_mask(img_rgb, cfg.COLOR_PALETTE['ROAD'],   cfg.COLOR_TOLERANCE)
    mask_plaza  = get_color_mask(img_rgb, cfg.COLOR_PALETTE['PLAZA'],  cfg.COLOR_TOLERANCE)
    mask_portal = get_color_mask(img_rgb, cfg.COLOR_PALETTE['PORTAL'], cfg.COLOR_TOLERANCE)
    mask_road = mask_road & (~mask_wall) & (~mask_portal) & (~mask_plaza)
    mask_plaza = mask_plaza & (~mask_wall) & (~mask_portal)
    mask_portal = remove_small_objects(mask_portal, min_size=30)                            #消除過小雜訊
    img_gray = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]

    car_road_radius = (6.0 / 2) / cfg.METERS_PER_PIXEL                                      #路寬篩選縫合
    mask_wide_road = binary_opening(mask_road, disk(car_road_radius))
    mask_alley = mask_road & (~mask_wide_road)                                              #只保留窄巷
    mask_valid_dest = mask_plaza | mask_alley
    dist_dest = distance_transform_edt(~mask_valid_dest)
    dist_red = distance_transform_edt(~mask_portal)
    mask_bridge = (dist_dest < 4.0) & (dist_red < 4.0) & (~mask_wall) & (~mask_valid_dest) & (~mask_portal)

    labeled_portals, num_portals = label(mask_portal, return_num=True, connectivity=2)      #斑馬線的權重計算
    props = regionprops(labeled_portals)
    penalty_dict = {}
    for p in props:
        length_m = p.major_axis_length * cfg.METERS_PER_PIXEL
        penalty_dict[p.label] = cfg.PENALTY_BASE + (length_m * cfg.PENALTY_FACTOR)          #等待時間=Base+(長度公尺*Factor)

    print(f"[{run_name}] 偵測到 {num_portals} 條傳送門")                                     #告訴你有幾條斑馬線，之後再針對起點附近的計算時間

    costs_start = np.full(img_rgb.shape[:2], 100000.0)                                      #偵測有效起點，不能在建築物和馬路裡面 
    costs_start[mask_plaza] = 1.0
    costs_start[mask_alley] = 1.0
    costs_start[mask_bridge] = 1.0
    mcp = MCP_Geometric(costs_start)                                                        #MCP_Geometric是skimage裡面的演算法，像在起點倒了一桶水
    cum_costs, _ = mcp.find_costs([cfg.START_POINT])                                        #每一個點累積了多少成本，大於1000會流不動
    
    if cum_costs[cfg.START_POINT[0], cfg.START_POINT[1]] > 1000:
        print(f" [{run_name}] 起點 ({cfg.START_POINT}) 落在無效區域，請調整起點位置後重試。")
        return None

    mask_start_rough = (cum_costs < 1000)                                                   #用1000為流動成本過濾出與起點相連的斑馬線
    labels_rough, _ = label(mask_start_rough, return_num=True, connectivity=1)
    start_id = labels_rough[cfg.START_POINT[0], cfg.START_POINT[1]]
    mask_start_strict = (labels_rough == start_id)              
    
    mask_dilated = binary_dilation(mask_start_strict, disk(4))                              #修正圖有雜質，斑馬線與廣場有縫隙連不起來的問題
    connected_ids = []
    for p in props:
        if np.any((labeled_portals == p.label) & mask_dilated):
            connected_ids.append(p.label)
            
    print(f"[{run_name}] 起點連接分支: {len(connected_ids)} 條")

    final_min_map = np.full(img_rgb.shape[:2], np.inf)              #開始分支模擬
    loop_ids = connected_ids if len(connected_ids) > 0 else []

    for idx, active_id in enumerate(loop_ids):    #開始用迴圈一條條算每條路徑的走法
        costs = np.full(img_rgb.shape[:2], 100000.0)
        costs[mask_plaza] = 1.0
        costs[mask_alley] = 1.0
        costs[mask_bridge] = 1.0
        
        for p in props:
            pid = p.label
            p_mask = (labeled_portals == pid)
            if pid == active_id:
                costs[p_mask] = penalty_dict[pid] #用三個IF/ELSE將選中的斑馬線(active_id)打開
            elif pid in connected_ids:
                costs[p_mask] = 100000.0          #同樣相連但不選擇的斑馬線關掉
            else:
                costs[p_mask] = penalty_dict[pid] #遠方的斑馬線還是可以繞遠路過去走，也打開
        
        mcp = MCP_Geometric(costs)                #設定好門的開關後，開始計算路徑需要的時間
        cum, _ = mcp.find_costs([cfg.START_POINT])
        t_map = cum / cfg.PIXELS_PER_MINUTE
        
        update = t_map < final_min_map            #如果走這個路徑所需時間比之前迴圈的還快，就更新final_min_map紀錄
        final_min_map[update] = t_map[update]
        
        plot_heatmap(img_gray, t_map, mask_portal, active_id,               #想看到每分支的圖都跑出來
                     f"{output_folder}/Branch_Portal_{active_id:02d}.png", 
                     title=f"{run_name}: Branch #{active_id}")

    plot_heatmap(img_gray, final_min_map, mask_portal, None,                #以最小時間的紀錄生成最終疊圖
                 f"{output_folder}/FINAL_Stacked.png", 
                 title=f"{run_name}: Final Stacked Analysis")
    
    return final_min_map

def plot_heatmap(img_gray, time_map, mask_portal, active_id, save_path, title):
    cfg = USER_CONFIG
    
    reachable = (time_map < 1000)
    vis_mask = binary_erosion(reachable, disk(2)) & (~mask_portal)
    vis_data = time_map.copy()                                                      #把走不到的地方挖空(nan),畫圖時變透明
    vis_data[~vis_mask] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_gray, cmap='gray', alpha=0.4)                                     #把底圖變成灰階，調整透明度
    
    vis_portals = np.zeros(img_gray.shape)
    vis_portals[mask_portal] = 1
    ax.imshow(vis_portals, cmap='spring', alpha=np.where(mask_portal, 0.6, 0.0))    #把紅色的斑馬線特別標示出來
    
    cmap = plt.cm.jet
    cmap.set_bad('none')                                                            #nan走不到的地方設為全透明
    im = ax.imshow(vis_data, cmap=cmap, alpha=0.6, vmin=0, vmax=cfg.MAX_VISUAL_TIME)#鎖定顏色範圍，藍色(冷/快)，紅色(熱/慢)
    
    if np.nanmax(vis_data) > 1:                                                     #如果有數據才畫，在我們指定分鐘(1/3/5...)的位置畫線
        CS = ax.contour(vis_data, levels=cfg.LEVELS, colors='white', linewidths=1.0, alpha=0.8)
        ax.clabel(CS, inline=True, fontsize=8, fmt='%.0f', colors='white')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Minutes', rotation=270, labelpad=10)
    
    ax.plot(cfg.START_POINT[1], cfg.START_POINT[0], 'w*', markersize=15, markeredgecolor='k')
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_comparison_map(map_before, map_after, save_path):                      #生成設計介入前後的差異圖(COMPARISON_Diff_Map.png)
    print("\n[Comparison] 生成差異分析圖")
    if map_before is None or map_after is None: 
        print("跳過對比圖：因為其中一張地圖生成失敗。")
        return

    t_before = map_before.copy()
    t_after = map_after.copy()
    
    mask_new_access = (t_before > 1000) & (t_after < 1000)                          #標記新可達區域
    
    diff = t_before - t_after
    mask_significant = (np.abs(diff) > 0.5) & (t_before < 1000) & (t_after < 1000)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(np.ones_like(t_before), cmap='gray')
    
    vis_diff = np.full(diff.shape, np.nan)
    vis_diff[mask_significant] = diff[mask_significant]
    
    im = ax.imshow(vis_diff, cmap=plt.cm.RdBu, vmin=-5, vmax=5, alpha=0.8)  #藍色改善，紅色變差
    
    vis_new = np.zeros((*diff.shape, 4))                                    #金色新可達區域
    vis_new[mask_new_access] = [1, 0.84, 0, 0.6] 
    ax.imshow(vis_new)
    
    start = USER_CONFIG.START_POINT
    ax.plot(start[1], start[0], 'k*', markersize=20)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Time Saved (min)\n(Blue = Faster)', rotation=270, labelpad=20)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='gold', edgecolor='none', label='New Access')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f"Design Benefit: Improvement Map", fontsize=16)
    ax.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"差異圖已儲存: {save_path}")

#執行區
#=========================================================================
if __name__ == "__main__":
    cfg = USER_CONFIG
    
    #現況分析
    matrix_existing = run_analysis_pipeline(cfg.FILE_EXISTING, "folder_existing", "EXISTING")
    
    #設計後分析
    matrix_proposed = run_analysis_pipeline(cfg.FILE_PROPOSED, "folder_proposed", "PROPOSED")
    
    #差異比較
    if matrix_existing is not None and matrix_proposed is not None:
        generate_comparison_map(matrix_existing, matrix_proposed, "COMPARISON_Diff_Map.png")
    
    print("\n 所有分析工作完成")
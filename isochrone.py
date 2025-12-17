import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.graph import MCP_Geometric
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_dilation
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import os

def generate_color_coded_analysis(image_path, start_point, save_folder="output_final_colors"):
    # ==========================================
    # 1. 參數設定
    # ==========================================
    pixels_per_minute = 100.0 
    meters_per_pixel = 0.8 
    
    # 阻力參數
    plaza_cost = 1.0          # 綠色區域阻力
    road_edge_cost = 1.0      # 灰色馬路(邊緣)阻力
    
    # 傳送門參數 (紅色斑馬線)
    base_penalty = 2.0        # 基礎罰時
    length_penalty_factor = 0.15 # 長度罰時係數
    
    # 智慧路寬參數 (針對灰色馬路)
    # Level 1: 超大馬路 (>18m) -> 整條封鎖
    huge_road_width_threshold_m = 18.0
    huge_radius_px = (huge_road_width_threshold_m / 2) / meters_per_pixel
    
    # Level 2: 一般馬路 -> 離路邊 >2m 的中心封鎖
    road_edge_walkable_width_m = 2.0
    edge_radius_px = road_edge_walkable_width_m / meters_per_pixel
    
    # 雜訊過濾 (紅色色塊太小視為雜訊)
    min_crosswalk_size = 30 
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f"處理圖片: {image_path}")
    print("模式: 指定顏色辨識 (綠=安, 黑=牆, 紅=門, 灰=路)")

    # ==========================================
    # 2. 精準顏色分類 (Color Segmentation)
    # ==========================================
    try:
        img_rgb = io.imread(image_path)
        if img_rgb.shape[-1] == 4: img_rgb = img_rgb[..., :3]
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {image_path}")
        return

    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    # 設定顏色容許值 (Tolerance)，避免 JPG 壓縮雜訊
    tol = 40 

    # A. 黑色 (建物) -> R, G, B 都很低
    mask_wall = (R < 50) & (G < 50) & (B < 50)
    
    # B. 綠色 (可行走) -> G 特別高
    mask_green_plaza = (G > 150) & (R < 150) & (B < 150)
    
    # C. 紅色 (斑馬線/傳送門) -> R 特別高
    mask_red_portal_raw = (R > 180) & (G < 100) & (B < 100)
    mask_red_portal = remove_small_objects(mask_red_portal_raw, min_size=min_crosswalk_size)
    
    # D. 灰色 (馬路) -> R, G, B 數值相近，且不是黑/白
    # 這裡我們用排除法：不是牆、不是綠、不是紅，剩下的就是路 (假設底圖乾淨)
    # 或者用更嚴謹的定義：
    mask_gray_road = (R > 50) & (R < 200) & \
                     (np.abs(R - G) < tol) & \
                     (np.abs(R - B) < tol) & \
                     (~mask_green_plaza) & (~mask_red_portal) & (~mask_wall)

    print("顏色偵測完成：")
    print(f"- 黑色(牆): {np.sum(mask_wall)} px")
    print(f"- 綠色(安): {np.sum(mask_green_plaza)} px")
    print(f"- 紅色(門): {np.sum(mask_red_portal)} px")
    print(f"- 灰色(路): {np.sum(mask_gray_road)} px")

    # ==========================================
    # 3. 針對「灰色區域」進行智慧路寬偵測
    # ==========================================
    print("正在分析道路寬度...")
    
    # Level 1: 偵測超大馬路 (整條封鎖)
    # 邏輯：灰色區域能塞進大圓盤的地方
    selem_huge = disk(huge_radius_px)
    mask_huge_road_core = binary_opening(mask_gray_road, selem_huge)
    # 膨脹回來，確保整條大路被標記
    mask_huge_road_blocked = binary_dilation(mask_huge_road_core, disk(edge_radius_px * 2))

    # Level 2: 偵測一般馬路 (中心封鎖)
    # 邏輯：灰色區域內部距離 > 2m 的地方
    road_dist_map = distance_transform_edt(mask_gray_road)
    mask_road_center_blocked = road_dist_map > edge_radius_px

    # 整合危險區域 (Danger Zone)
    # 這些地方之後會被設為無限大阻力
    mask_danger_zone = (mask_huge_road_blocked | mask_road_center_blocked)
    
    # 安全檢查：確保危險區不會覆蓋到紅色斑馬線或綠色廣場
    mask_danger_zone = mask_danger_zone & (~mask_red_portal) & (~mask_green_plaza)

    # ==========================================
    # 4. 分析紅色傳送門
    # ==========================================
    labeled_portals, num_portals = label(mask_red_portal, return_num=True, connectivity=2)
    props = regionprops(labeled_portals)
    
    penalty_dict = {}
    print(f"-> 偵測到 {num_portals} 條紅色傳送門")
    
    for prop in props:
        # 計算長度 (公尺)
        real_length_m = prop.major_axis_length * meters_per_pixel
        # 計算罰時
        penalty_cost = base_penalty + (real_length_m * length_penalty_factor)
        penalty_dict[prop.label] = penalty_cost

    # ==========================================
    # 5. 批次產圖迴圈
    # ==========================================
    final_min_time_map = np.full(img_rgb.shape[:2], np.inf)
    time_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20]
    
    # 準備顯示用的底圖 (灰階化)
    img_gray_display = 0.299 * R + 0.587 * G + 0.114 * B

    # 迴圈範圍：如果沒有偵測到門，至少跑一次(看純走路)
    loop_range = range(1, num_portals + 1) if num_portals > 0 else [0]

    for i in loop_range:
        # 建立 Cost Map (預設全通)
        costs = np.ones(img_rgb.shape[:2], dtype=float)
        
        # 1. 綠色區域 (好走)
        costs[mask_green_plaza] = plaza_cost
        
        # 2. 灰色區域 (預設好走，但馬上要蓋上危險區)
        costs[mask_gray_road] = road_edge_cost
        
        # 3. 封鎖牆壁 & 危險道路(大路全封/中路封心)
        costs[mask_wall] = 100000
        costs[mask_danger_zone] = 100000
        
        # 4. 傳送門邏輯
        if num_portals > 0:
            # 開啟當前門
            current_portal_mask = (labeled_portals == i)
            this_penalty = penalty_dict.get(i, 5.0)
            costs[current_portal_mask] = this_penalty
            
            # 關閉其他門 (阻斷)
            other_portals_mask = (mask_red_portal) & (~current_portal_mask)
            costs[other_portals_mask] = 100000 
        
        # 5. 執行路徑分析
        mcp = MCP_Geometric(costs)
        cumulative_costs, traceback = mcp.find_costs([start_point])
        time_map = cumulative_costs / pixels_per_minute
        
        # 6. 更新最終疊圖
        update_mask = time_map < final_min_time_map
        final_min_time_map[update_mask] = time_map[update_mask]

        # --- 單張圖視覺化 ---
        if num_portals > 0:
            # 遮罩：牆壁 | 危險路 | 其他門 | 當前門(不顯示漸層)
            visual_mask = mask_wall | (mask_danger_zone & ~current_portal_mask) | other_portals_mask | current_portal_mask
            time_map_visual = time_map.copy()
            time_map_visual[visual_mask] = np.nan
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img_gray_display, cmap='gray', alpha=0.4)
            
            # 顯示被封鎖的馬路 (淡紅色) - 供檢查用
            blocked_road_vis = np.zeros(img_rgb.shape[:2])
            blocked_road_vis[mask_danger_zone] = 1
            ax.imshow(blocked_road_vis, cmap='Reds', alpha=np.where(mask_danger_zone, 0.2, 0.0))

            # 顯示當前傳送門 (亮粉色塊)
            portal_vis = np.zeros(img_rgb.shape[:2])
            portal_vis[current_portal_mask] = 1
            ax.imshow(portal_vis, cmap='spring', alpha=np.where(current_portal_mask, 0.8, 0.0))

            # 熱力圖
            cmap = plt.cm.jet
            cmap.set_bad(color='black', alpha=0)
            ax.imshow(time_map_visual, cmap=cmap, alpha=0.6, vmin=0, vmax=20)
            
            # 等時線
            if np.nanmax(time_map_visual) > time_levels[0]:
                CS = ax.contour(time_map_visual, levels=time_levels, colors='white', linewidths=1.0, alpha=0.8)
                ax.clabel(CS, inline=True, fontsize=8, fmt='%1.0f', colors='white')
                
            ax.plot(start_point[1], start_point[0], 'w*', markersize=15, markeredgecolor='k')
            ax.set_title(f"Portal #{i} Active (Cost: {this_penalty*0.5:.1f} min)", fontsize=12)
            ax.axis('off')
            
            plt.savefig(f"{save_folder}/portal_{i:02d}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    # ==========================================
    # 6. 生成最終合成圖
    # ==========================================
    print("生成最終疊圖...")
    
    # 最終遮罩：牆壁 | 危險路 | 所有門(統一單色顯示)
    final_mask = mask_wall | (mask_danger_zone & ~mask_red_portal) | mask_red_portal
    final_min_time_map[final_mask] = np.nan
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_gray_display, cmap='gray', alpha=0.4)
    
    # 顯示危險路寬 (淡紅)
    danger_vis = np.zeros(img_rgb.shape[:2])
    danger_vis[mask_danger_zone] = 1
    ax.imshow(danger_vis, cmap='Reds', alpha=np.where(mask_danger_zone, 0.2, 0.0))

    # 顯示所有傳送門 (粉色)
    all_portals_vis = np.zeros(img_rgb.shape[:2])
    all_portals_vis[mask_red_portal] = 1
    ax.imshow(all_portals_vis, cmap='spring', alpha=np.where(mask_red_portal, 0.6, 0.0))

    # 熱力圖
    cmap = plt.cm.jet
    cmap.set_bad(color='black', alpha=0)
    im = ax.imshow(final_min_time_map, cmap=cmap, alpha=0.6, vmin=0, vmax=20)
    
    if np.nanmax(final_min_time_map) > time_levels[0]:
        CS = ax.contour(final_min_time_map, levels=time_levels, colors='white', linewidths=1.5, alpha=0.9)
        ax.clabel(CS, inline=True, fontsize=10, fmt='%1.0f min', colors='white')
        
    ax.plot(start_point[1], start_point[0], 'w*', markersize=18, markeredgecolor='k', label='Start')
    ax.set_title("Final Analysis: Color Coded (Green/Gray/Red)", fontsize=14)
    ax.axis('off')
    
    plt.savefig(f"{save_folder}/FINAL_Color_Overlay.png", dpi=300, bbox_inches='tight')
    print(f"完成！請檢查 {save_folder}")

# ==========================================
# 執行區
# ==========================================
# 請確認你的檔名是這個
file_existing = "site_existing.jpg" 
start_y = 500  # 請記得根據新圖調整起點座標！
start_x = 1000

generate_color_coded_analysis(
    file_existing, 
    (start_y, start_x), 
    save_folder="output_final_colors"
)
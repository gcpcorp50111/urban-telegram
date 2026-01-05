import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.graph import MCP_Geometric
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_erosion, binary_dilation
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import os

def generate_gradient_branching(image_path, start_point, save_folder="output_gradient"):
    # ==========================================
    # 1. 參數設定
    # ==========================================
    pixels_per_minute = 100.0  # 根據您的圖面比例調整
    meters_per_pixel = 0.8 
    
    plaza_cost = 1.0          
    alley_cost = 1.0
    
    # 這裡設定較低的罰時，讓熱力圖能跑遠一點
    base_penalty = 0.5        
    length_penalty_factor = 0.06 
    
    # 【關鍵視覺設定】
    # 設定熱力圖的「絕對上限」，超過這個時間顯示為深紅/黑色
    # 設定為 30 分鐘，這樣 1~10 分鐘的層次會很明顯
    max_visual_time = 30.0 
    
    proximity_threshold_px = 4.0
    car_road_width_threshold_m = 6.0
    car_road_radius_px = (car_road_width_threshold_m / 2) / meters_per_pixel
    min_crosswalk_size = 30 
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f"處理圖片: {image_path}")
    print("模式: 全彩漸層熱力圖 (Gradient Visualization)")

    # ==========================================
    # 2. 顏色分類 (嚴格模式)
    # ==========================================
    try:
        img_rgb = io.imread(image_path)
        if img_rgb.shape[-1] == 4: img_rgb = img_rgb[..., :3]
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {image_path}")
        return

    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    # 嚴格色票判定
    mask_wall = (R < 80) & (G < 80) & (B < 80)
    mask_red_raw = (R > 180) & (G < 150) & (B < 150)
    mask_red = remove_small_objects(mask_red_raw, min_size=min_crosswalk_size)
    mask_green = (G > 180) & (R < 180) & (B < 180)
    mask_white_potential = (R > 180) & (G > 180) & (B > 180)
    mask_white_potential = mask_white_potential & (~mask_red) & (~mask_green) & (~mask_wall)

    img_gray_display = 0.299 * R + 0.587 * G + 0.114 * B

    # ==========================================
    # 3. 路寬 & 縫合
    # ==========================================
    selem = disk(car_road_radius_px)
    mask_wide_road = binary_opening(mask_white_potential, selem)
    mask_alley = mask_white_potential & (~mask_wide_road)
    
    mask_valid_destinations = mask_green | mask_alley
    dist_to_dest = distance_transform_edt(~mask_valid_destinations)
    dist_to_red = distance_transform_edt(~mask_red)
    mask_bridge_pixels = (dist_to_dest < proximity_threshold_px) & \
                         (dist_to_red < proximity_threshold_px) & \
                         (~mask_wall) & (~mask_valid_destinations) & (~mask_red)

    # ==========================================
    # 4. 分析傳送門
    # ==========================================
    labeled_portals, num_portals = label(mask_red, return_num=True, connectivity=2)
    props = regionprops(labeled_portals)
    
    penalty_dict = {}
    for prop in props:
        real_length_m = prop.major_axis_length * meters_per_pixel
        new_cost = base_penalty + (real_length_m * length_penalty_factor)
        penalty_dict[prop.label] = new_cost
    
    # ==========================================
    # 5. 起點腹地偵測
    # ==========================================
    print("偵測起點腹地...")
    costs_start_only = np.full(img_rgb.shape[:2], 100000.0)
    costs_start_only[mask_green] = plaza_cost
    costs_start_only[mask_alley] = alley_cost
    costs_start_only[mask_bridge_pixels] = plaza_cost
    
    mcp_start = MCP_Geometric(costs_start_only)
    cum_costs_start, _ = mcp_start.find_costs([start_point])
    
    mask_start_zone_rough = (cum_costs_start < 1000)
    labels_start, _ = label(mask_start_zone_rough, return_num=True, connectivity=1)
    start_label_id = labels_start[start_point[0], start_point[1]]
    
    if start_label_id == 0:
        print("錯誤：起點落在無效區域！")
        return

    mask_start_zone_strict = (labels_start == start_label_id)
    mask_start_zone_dilated = binary_dilation(mask_start_zone_strict, disk(4))
    
    start_connected_portal_ids = []
    for prop in props:
        if np.any((labeled_portals == prop.label) & mask_start_zone_dilated):
            start_connected_portal_ids.append(prop.label)
            
    print(f"-> 發現 {len(start_connected_portal_ids)} 條起點分支路徑")

    # ==========================================
    # 6. 分支模擬迴圈 (繪製漸層圖)
    # ==========================================
    final_min_time_map = np.full(img_rgb.shape[:2], np.inf)
    
    # 設定等高線 (分鐘)
    time_levels = [2, 5, 8, 12, 16, 20, 25]

    for idx, active_portal_id in enumerate(start_connected_portal_ids):
        print(f"  繪製分支 {idx+1}/{len(start_connected_portal_ids)}: Portal #{active_portal_id}")
        
        costs = np.full(img_rgb.shape[:2], 100000.0)
        costs[mask_green] = plaza_cost
        costs[mask_alley] = alley_cost
        costs[mask_bridge_pixels] = plaza_cost
        
        for prop in props:
            p_id = prop.label
            p_mask = (labeled_portals == p_id)
            p_cost = penalty_dict[p_id]
            
            if p_id == active_portal_id:
                costs[p_mask] = p_cost
            elif p_id in start_connected_portal_ids:
                costs[p_mask] = 100000.0
            else:
                costs[p_mask] = p_cost
        
        mcp = MCP_Geometric(costs)
        cumulative_costs, _ = mcp.find_costs([start_point])
        time_map = cumulative_costs / pixels_per_minute
        
        # 更新疊圖數據
        update_mask = time_map < final_min_time_map
        final_min_time_map[update_mask] = time_map[update_mask]
        
        # --- 單張圖視覺化 ---
        reachable_mask = (time_map < 1000)
        selem_erode = disk(2)
        eroded_reachable_mask = binary_erosion(reachable_mask, selem_erode)
        
        # 視覺遮罩：只顯示可達區域，且扣掉傳送門本身
        vis_mask = eroded_reachable_mask & (~mask_red) & (~mask_bridge_pixels)
        time_map_vis = time_map.copy()
        time_map_vis[~vis_mask] = np.nan
        
        fig, ax = plt.subplots(figsize=(10, 10))
        # 底圖
        ax.imshow(img_gray_display, cmap='gray', alpha=0.4)
        
        # 標示「被選中」的門 (白色邊框 + 亮粉色)
        active_mask = (labeled_portals == active_portal_id)
        active_vis = np.zeros(img_rgb.shape[:2])
        active_vis[active_mask] = 1
        ax.imshow(active_vis, cmap='spring', alpha=np.where(active_mask, 1.0, 0.0))
        
        # 【重點】繪製漸層熱力圖 (Jet Colormap)
        # vmin=0 (藍色), vmax=30 (紅色)
        cmap = plt.cm.jet
        cmap.set_bad(color='none') # NaN 設為透明
        im = ax.imshow(time_map_vis, cmap=cmap, alpha=0.6, vmin=0, vmax=max_visual_time)
        
        # 繪製等高線
        if np.nanmax(time_map_vis) > time_levels[0]:
            CS = ax.contour(time_map_vis, levels=time_levels, colors='white', linewidths=0.8, alpha=0.5)
            ax.clabel(CS, inline=True, fontsize=8, fmt='%1.0f min', colors='white')

        # 加入 Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Walking Time (min)', rotation=270, labelpad=15)
            
        ax.plot(start_point[1], start_point[0], 'w*', markersize=15, markeredgecolor='k')
        ax.set_title(f"Scenario: Through Portal #{active_portal_id}", fontsize=12)
        ax.axis('off')
        
        plt.savefig(f"{save_folder}/Gradient_Scenario_{active_portal_id:02d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ==========================================
    # 7. 生成最終疊圖
    # ==========================================
    print("生成最終疊圖...")
    
    reachable_mask = (final_min_time_map < 1000)
    selem_erode = disk(2)
    eroded_reachable_mask = binary_erosion(reachable_mask, selem_erode)
    
    final_vis_mask = eroded_reachable_mask & (~mask_red) & (~mask_bridge_pixels)
    final_time_vis = final_min_time_map.copy()
    final_time_vis[~final_vis_mask] = np.nan
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_gray_display, cmap='gray', alpha=0.4)
    
    # 顯示所有紅門
    all_portals_vis = np.zeros(img_rgb.shape[:2])
    all_portals_vis[mask_red] = 1
    ax.imshow(all_portals_vis, cmap='spring', alpha=np.where(mask_red, 0.6, 0.0))
    
    # 漸層熱力圖
    cmap = plt.cm.jet
    cmap.set_bad(color='none')
    im = ax.imshow(final_time_vis, cmap=cmap, alpha=0.6, vmin=0, vmax=max_visual_time)
    
    if np.nanmax(final_time_vis) > time_levels[0]:
        CS = ax.contour(final_time_vis, levels=time_levels, colors='white', linewidths=1.2, alpha=0.8)
        ax.clabel(CS, inline=True, fontsize=10, fmt='%1.0f min', colors='white')
        
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Walking Time (min)', rotation=270, labelpad=15)
        
    ax.plot(start_point[1], start_point[0], 'w*', markersize=18, markeredgecolor='k', label='Start')
    ax.set_title("Final Analysis: Gradient Overlay", fontsize=14)
    ax.axis('off')
    
    plt.savefig(f"{save_folder}/FINAL_Gradient_Stack.png", dpi=300, bbox_inches='tight')
    print(f"完成！請查看 {save_folder}")

# ==========================================
# 執行區
# ==========================================
file_existing = "site_existing.jpg" 
start_y = 500 
start_x = 1100

generate_gradient_branching(
    file_existing, 
    (start_y, start_x), 
    save_folder="output_gradient"
)
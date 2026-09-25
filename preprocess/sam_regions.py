# Adapted from OmniPart, Copyright (c) 2025 VAST-AI-Research and contributors.
# See LICENSE-OMNIPART. Visualization and batch orchestration are omitted.
import numpy as np
import cv2
size_th = 1000

def split_disconnected_parts(group_ids, size_threshold=None):
    """
    Split each part into separate parts if they contain disconnected regions.
    
    Args:
        group_ids: Array of segment IDs for each pixel
        size_threshold: Minimum size threshold for considering a segment (in pixels).
                       If None, uses the global size_th variable.
        
    Returns:
        Updated array with each connected component having a unique ID
    """
    # Use provided threshold or fall back to global variable
    if size_threshold is None:
        size_threshold = size_th
    # Create a copy to hold the result
    new_group_ids = np.full_like(group_ids, -1)  # Start with all background
    
    # Get unique part IDs (excluding background -1)
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    
    # Track the next available ID
    next_id = 0
    total_split_regions = 0
    
    # For each existing part ID
    for part_id in unique_ids:
        # Extract the mask for this part
        part_mask = (group_ids == part_id).astype(np.uint8)
        
        # Find connected components within this part
        num_labels, labels = cv2.connectedComponents(part_mask, connectivity=8)
        
        if num_labels == 1:  # Just background (0), no regions found
            continue
            
        if num_labels == 2:  # One connected component (background + 1 region)
            # Assign the original part's area to the next available ID
            new_group_ids[labels == 1] = next_id
            next_id += 1
        else:  # Multiple disconnected components
            split_count = 0
            print(f"Part {part_id} has {num_labels-1} disconnected regions, splitting...")
            
            # For each connected component (skipping background label 0)
            for label in range(1, num_labels):
                region_mask = labels == label
                region_size = np.sum(region_mask)
                
                # Only include regions that are large enough
                if region_size >= size_threshold / 5:  # Using size threshold to avoid tiny fragments
                    new_group_ids[region_mask] = next_id
                    split_count += 1
                    next_id += 1
                else:
                    print(f"  Skipping small disconnected region ({region_size} pixels)")
            
            total_split_regions += split_count
            
    if total_split_regions > 0:
        print(f"Split disconnected parts: original {len(unique_ids)} parts -> {next_id} connected parts")
    else:
        print("No parts needed splitting - all parts are already connected")
    
    return new_group_ids

# -------------------------------------------------------
# MAIN SEGMENTATION FUNCTION
# -------------------------------------------------------

def get_sam_mask(image, mask_generator, visual, merge_groups=None, existing_group_ids=None, 
                check_undetected=True, rgba_image=None, img_name=None, skip_split=False, save_dir=None, size_threshold=None):
    """
    Generate and process SAM masks for the image, with optional merging and undetected region detection.
    
    Args:
        size_threshold: Minimum size threshold for considering a segment (in pixels). 
                       If None, uses the global size_th variable.
    """
    # Use provided threshold or fall back to global variable
    if size_threshold is None:
        size_threshold = size_th
    label_mode = '1'
    anno_mode = ['Mask', 'Mark']
    
    exist_group = False

    # Use existing group IDs if provided, otherwise generate new ones with SAM
    if existing_group_ids is not None:
        group_ids = existing_group_ids.copy()
        group_counter = np.max(group_ids) + 1
        exist_group = True
    else:
        # Generate masks using SAM
        masks = mask_generator.generate(image)
        group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
        num_masks = len(masks)
        group_counter = 0

        # Sort masks by area (largest first)
        area_sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        
        # Create background mask if we have RGBA image
        background_mask = None
        if rgba_image is not None:
            rgba_array = np.array(rgba_image)
            if rgba_array.shape[2] == 4:
                # Use alpha channel to create foreground/background mask
                background_mask = rgba_array[:, :, 3] <= 10  # Areas with very low alpha are background

        # First pass: assign original group IDs
        for i in range(0, num_masks):
            if area_sorted_masks[i]["area"] < size_threshold:
                print(f"Skipping mask {i}, area too small: {area_sorted_masks[i]['area']} < {size_threshold}")
                continue
            
            mask = area_sorted_masks[i]["segmentation"]
            
            # Check proportion of background pixels in this mask
            if background_mask is not None:
                # Calculate how many pixels in this mask are background
                background_pixels_in_mask = np.sum(mask & background_mask)
                mask_area = np.sum(mask)
                background_ratio = background_pixels_in_mask / mask_area
                
                # Skip mask if background proportion is too high (>10%)
                if background_ratio > 0.1:
                    print(f"  Skipping mask {i}, background ratio: {background_ratio:.2f}")
                    continue
            
            # Assign group ID to this mask's pixels
            group_ids[mask] = group_counter
            print(f"Assigned mask {i} with area {area_sorted_masks[i]['area']} to group {group_counter}")
            group_counter += 1
        
        # Split disconnected parts immediately after SAM segmentation
        print("Splitting disconnected parts in initial segmentation...")
        group_ids = split_disconnected_parts(group_ids, size_threshold)
        
        # Update group counter after splitting
        if np.max(group_ids) >= 0:
            group_counter = np.max(group_ids) + 1
        print(f"After early splitting, now have {len(np.unique(group_ids))-1} regions (excluding background)")
    
    # Check for undetected parts using RGBA information
    if check_undetected and rgba_image is not None:
        print("Checking for undetected parts using RGBA image...")
        # Create a foreground mask from the alpha channel
        rgba_array = np.array(rgba_image)
        
        # Check if the image has an alpha channel
        if rgba_array.shape[2] == 4:
            print(f"Image has alpha channel, checking for undetected parts...")
            # Use alpha channel to identify non-transparent pixels (foreground)
            alpha_mask = rgba_array[:, :, 3] > 0  
            
            # Create existing parts mask and dilate it
            existing_parts_mask = (group_ids != -1)
            kernel = np.ones((4, 4), np.uint8)
            
            # Use larger kernel for faster dilation
            large_kernel = np.ones((4, 4), np.uint8)
            dilated_parts = cv2.dilate(existing_parts_mask.astype(np.uint8), large_kernel)
            
            # Find undetected areas (foreground but not detected by SAM)
            undetected_mask = alpha_mask & (~dilated_parts.astype(bool))
            
            # Process only if there are enough undetected pixels
            if np.sum(undetected_mask) > size_threshold:
                print(f"Found undetected parts with {np.sum(undetected_mask)} pixels")
                
                # Find connected components in undetected regions
                num_labels, labels = cv2.connectedComponents(
                    undetected_mask.astype(np.uint8), 
                    connectivity=8
                )
                
                print(f"  Found {num_labels-1} initial regions")
                
                # Use Union-Find data structure for efficient region merging
                parent = list(range(num_labels))
                
                # Find with path compression
                def find(x):
                    """Find with path compression for Union-Find"""
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                
                # Union by rank/size
                def union(x, y):
                    """Union operation for Union-Find"""
                    root_x = find(x)
                    root_y = find(y)
                    if root_x != root_y:
                        # Use smaller ID as parent
                        if root_x < root_y:
                            parent[root_y] = root_x
                        else:
                            parent[root_x] = root_y
                
                # Calculate areas for all regions at once
                areas = np.bincount(labels.flatten())[1:] if num_labels > 1 else []
                
                # Filter regions by minimum size
                valid_regions = np.where(areas >= size_threshold/5)[0] + 1
                
                # Barrier mask for connectivity checks
                barrier_mask = existing_parts_mask
                
                # Pre-compute dilated regions for all valid regions
                dilated_regions = {}
                for i in valid_regions:
                    region_mask = (labels == i).astype(np.uint8)
                    dilated_regions[i] = cv2.dilate(region_mask, kernel, iterations=2)
                
                # Check for region merges based on proximity and overlap
                for idx, i in enumerate(valid_regions[:-1]):
                    for j in valid_regions[idx+1:]:
                        # Check overlap between dilated regions
                        overlap = dilated_regions[i] & dilated_regions[j]
                        overlap_size = np.sum(overlap)
                        
                        # Merge if significant overlap and not separated by existing parts
                        if overlap_size > 40 and not np.any(overlap & barrier_mask):
                            # Calculate overlap ratios
                            overlap_ratio_i = overlap_size / areas[i-1]
                            overlap_ratio_j = overlap_size / areas[j-1]
                            
                            if max(overlap_ratio_i, overlap_ratio_j) > 0.03:
                                union(i, j)
                                print(f"  Merging regions {i} and {j} (overlap: {overlap_size} px)")
                
                # Apply the merging results to create merged labels
                merged_labels = np.zeros_like(labels)
                for label in range(1, num_labels):
                    merged_labels[labels == label] = find(label)
                
                # Get unique merged regions
                unique_merged_regions = np.unique(merged_labels[merged_labels > 0])
                print(f"  After merging: {len(unique_merged_regions)} connected regions")
                
                # Add regions to group_ids if they're large enough
                group_counter_start = group_counter
                for label in unique_merged_regions:
                    region_mask = merged_labels == label
                    region_size = np.sum(region_mask)
                    
                    if region_size > size_threshold:
                        print(f"  Adding region with ID {label} ({region_size} pixels) as group {group_counter}")
                        group_ids[region_mask] = group_counter
                        group_counter += 1
                    else:
                        print(f"  Skipping small region with ID {label} ({region_size} pixels < {size_threshold})")
                
                print(f"  Added {group_counter - group_counter_start} regions that weren't detected by SAM")

                # Process edges for all new parts at once
                if group_counter > group_counter_start:
                    print("Processing edges for newly detected parts...")
                    
                    # Create combined mask for all new parts
                    new_parts_mask = np.zeros_like(group_ids, dtype=bool)
                    for part_id in range(group_counter_start, group_counter):
                        new_parts_mask |= (group_ids == part_id)
                    
                    # Compute edges for all new parts at once
                    all_new_dilated = cv2.dilate(new_parts_mask.astype(np.uint8), kernel, iterations=1)
                    all_new_eroded = cv2.erode(new_parts_mask.astype(np.uint8), kernel, iterations=1)
                    all_new_edges = all_new_dilated.astype(bool) & (~all_new_eroded.astype(bool))
                    
                    print(f"Edge processing completed for {group_counter - group_counter_start} new parts")

    # Save debug visualization of initial segmentation
    if not exist_group:
        pass

    # Merge groups if specified
    if merge_groups is not None:
        # Start with current group_ids
        merged_group_ids = group_ids
        
        # Preserve background regions
        merged_group_ids[group_ids == -1] = -1
        
        # For each merge group, assign all pixels to the first ID in that group
        for new_id, group in enumerate(merge_groups):
            # Create a mask to include all original IDs in this group
            group_mask = np.zeros_like(group_ids, dtype=bool)

            orig_ids_first = group[0]
            # Process each original ID
            for orig_id in group:
                # Get mask for this original ID
                mask = (group_ids == orig_id)
                pixels = np.sum(mask)
                if pixels > 0:
                    print(f"  Including original ID {orig_id} ({pixels} pixels)")
                    group_mask = group_mask | mask
                else:
                    print(f"  Warning: Original ID {orig_id} does not exist")
            
            # Set all pixels in this group to the first ID in the group
            if np.any(group_mask):
                print(f"  Merging {np.sum(group_mask)} pixels to ID {orig_ids_first}")
                merged_group_ids[group_mask] = orig_ids_first

        # Reassign IDs to be continuous from 0
        unique_ids = np.unique(merged_group_ids)
        unique_ids = unique_ids[unique_ids != -1]  # Exclude background
        id_reassignment = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

        # Create new array with reassigned IDs
        new_group_ids = np.full_like(merged_group_ids, -1)  # Start with all background
        for old_id, new_id in id_reassignment.items():
            new_group_ids[merged_group_ids == old_id] = new_id

        # Update merged_group_ids with continuous IDs
        merged_group_ids = new_group_ids

        print(f"ID reassignment complete: {len(id_reassignment)} groups now have sequential IDs from 0 to {len(id_reassignment)-1}")

        # Replace original group IDs with merged result
        group_ids = merged_group_ids
        print(f"Merging complete, now have {len(np.unique(group_ids))-1} regions (excluding background)")
        
        # Skip splitting disconnected parts if requested
        if not skip_split:
            # Split disconnected parts into separate parts
            group_ids = split_disconnected_parts(group_ids, size_threshold)
            print(f"After splitting disconnected parts, now have {len(np.unique(group_ids))-1} regions (excluding background)")
    else:
        # Always split disconnected parts for initial segmentation
        group_ids = split_disconnected_parts(group_ids, size_threshold)
        print(f"After splitting disconnected parts, now have {len(np.unique(group_ids))-1} regions (excluding background)")

    return group_ids

classdef ExperimentSeed
    
    properties
        original_image_path
        processed_image_path
        predictions_path
        res_path
        configs_path
        prediction_entry_path
        norm_type
        interp_type
        configs
        original_images = []
        processed_images = []
        original_results = []
        processed_results = []
        n_organs = 5;
        global_slice_thickness = 1.2; % mm
        global_fov = 150; % mm
        vol_size_orig = []
        vol_crop_size = []
        top_remove = []
        bottom_remove = []
        left_remove = []
        right_remove = []
    end
    
    methods
        function experiment = ExperimentSeed(original_image_path, processed_image_path, res_path, predictions_path, configs_path, prediction_entry_path, norm_type, interp_type, configs)
            experiment.original_image_path = original_image_path;
            experiment.processed_image_path = processed_image_path;
            experiment.res_path = res_path;
            experiment.predictions_path = predictions_path;
            experiment.configs_path = configs_path;
            experiment.prediction_entry_path = prediction_entry_path;
            experiment.norm_type = norm_type;
            experiment.interp_type = interp_type;
            experiment.configs = configs;
        end
        
        function experiment = run(experiment)
            experiment = experiment.prepare_images();
            experiment = experiment.prepare_configs();
            experiment = experiment.process_results();
            experiment.delete_images()
            experiment.delete_configs()
        end
    end
    
    methods
        function experiment = prepare_images(experiment)            
            hinfo = hdf5info(experiment.original_image_path);
            n_datasets = length(hinfo.GroupHierarchy.Datasets);
            for k=1:n_datasets
                if k == 1
                    row_res = h5read(experiment.res_path,'/row_res');
                    slice_res = h5read(experiment.res_path,'/slice_res');
                    col_res = row_res;
                end
                new_img = h5read(experiment.original_image_path,['/' num2str(k-1)]);
                vol(:,:,k) = permute(new_img,[3,2,1]);
            end
            experiment.original_images = vol;
            size(vol)

            if strcmp(experiment.interp_type,'naive')
                % naive resize
                vol = uint16(vol);
                experiment.vol_size_orig = size(vol);
                vol = imresize(vol, [512, 512], 'bicubic');
                vol_size = size(vol);
            elseif strcmp(experiment.interp_type,'same_fov')
                % interpolate such that in-plane spatial resolution
                % is 1/4 the slice thickness
                vol = uint16(vol);
                experiment.vol_size_orig = size(vol);
                size(vol)

                fov_r = experiment.vol_size_orig(1) * spatial_res(1);
                fov_c = experiment.vol_size_orig(2) * spatial_res(2);
                fov_s = experiment.vol_size_orig(3) * spatial_res(3);
                vol = permute(vol, [1,3,2]);
                size(vol)
                vol = imresize(vol, [experiment.vol_size_orig(1), round(fov_s/experiment.global_slice_thickness)], 'bicubic');
                size(vol)
                vol = permute(vol, [1,3,2]);
                size(vol)
                n_remove_r = (fov_r - experiment.global_fov) / spatial_res(1);
                n_remove_c = (fov_c - experiment.global_fov) / spatial_res(2);
                if n_remove_r < 0
                    n_remove_r = 0;
                end
                if n_remove_c < 0
                    n_remove_c = 0;
                end
                size(vol)
                experiment.top_remove = ceil(n_remove_r/2);
                experiment.bottom_remove = ceil(n_remove_r/2);
                experiment.left_remove = ceil(n_remove_c/2);
                experiment.right_remove = ceil(n_remove_c/2);
                vol = vol(experiment.top_remove+1:end-experiment.bottom_remove,...
                               experiment.left_remove+1:end-experiment.right_remove,...
                               :);
                size(vol)
                experiment.vol_crop_size = size(vol);
                size(vol)
                vol = imresize(vol, [512, 512], 'bicubic');
                vol_size = size(vol);
            else
                error('Invalid interpolation type.')
            end

            if strcmp(experiment.norm_type,'none')
                vol = vol;
            elseif strcmp(experiment.norm_type, 'unity')
                for k=1:vol_size(3)
                    img_s = double(vol(:,:,k));
                    img_s = img_s / max(img_s(:));
                    imgs(:,:,k) = img_s;
                end
                vol = imgs;
            elseif strcmp(experiment.norm_type,'mn3std')
                for k=1:vol_size(3)
                    img_s = double(vol(:,:,k));
                    mean_img = mean(img_s(:));
                    std_img = std(img_s(:));
                    min_val = mean_img - 3*std_img;
                    max_val = mean_img + 3*std_img;
                    idx = img_s < min_val;
                    img_s(idx) = min_val;
                    idx = img_s > max_val;
                    img_s(idx) = max_val;
                    img_s = (img_s - mean_img) / (3*std_img);
                    img_s = img_s - min(img_s(:));
                    imgs(:,:,k) = img_s;
                end
                vol = imgs;
            elseif strcmp(experiment.norm_type,'mn3std_wn')
                for k=1:vol_size(3)
                    img_s = double(vol(:,:,k));
                    mean_img = mean(img_s(:));
                    std_img = std(img_s(:));
                    min_val = mean_img - 3*std_img;
                    max_val = mean_img + 3*std_img;
                    idx = img_s < min_val;
                    img_s(idx) = min_val;
                    idx = img_s > max_val;
                    img_s(idx) = max_val;
                    img_s = (img_s - mean_img) / (3*std_img);
                    img_s = img_s - min(img_s(:));
                    mn = mean(img_s(:));
                    min_ = min(img_s(:));
                    max_ = max(img_s(:));
                    idx1 = img_s == min_;
                    idx2 = img_s == max_;
                    idx3 = idx1 | idx2;
                    inds = find(idx3>0);
                    for l=1:length(inds)
                        img_s(inds(l)) = mn*rand();
                    end
                    imgs(:,:,k) = img_s;
                end
                vol = imgs;
            else
                error('Invalid normalization type.')
            end
            
            experiment.processed_images = vol;

            for k=1:vol_size(3)
                if k==1
                    hdf5write(experiment.processed_image_path, ['/' num2str(k-1)], permute(vol(:, :, k),[3,2,1]));
                else
                    hdf5write(experiment.processed_image_path, ['/' num2str(k-1)], permute(vol(:, :, k),[3,2,1]),...
                                     'WriteMode', 'append');
                end
            end
        end
        
        function experiment = prepare_configs(experiment)
            experiment.configs.paths.test_X = experiment.processed_image_path;
        end
        
        function experiment = process_results(experiment)
            config_to_save = jsonencode(experiment.configs);
            fid = fopen(experiment.configs_path, 'w');
            fwrite(fid, config_to_save, 'char')
            fclose(fid);
            
            command = ['python "' experiment.prediction_entry_path '" "' experiment.configs_path '"'];            
            system(command);

            pred_fn = experiment.predictions_path;
            hinfo = hdf5info(pred_fn);
            original_preds = h5read(pred_fn, hinfo.GroupHierarchy.Datasets.Name);
            experiment.original_results = original_preds;
            clean_preds = zeros(size(experiment.processed_images));
            clean_preds_rz = zeros(size(experiment.original_images));
            
            [~, argmax_preds] = max(original_preds, [], 1);
            argmax_preds = squeeze(argmax_preds) - 1;
            preds = argmax_preds;
            preds = permute(preds, [2,1,3]);
            experiment.original_results = permute(experiment.original_results,[3,2,4,1]);
            for l=1:experiment.n_organs
                pred = preds == l;
                if sum(preds(:)) > 0
                    con = bwconncomp(pred);
                    props = regionprops3(con, 'Volume');
                    [~, idx] = sort(props.Volume, 'descend');

                    % assume EUS, SV, bladder, & rectum form closed surfaces
                    if ~isempty(idx)
                        pixels = con.PixelIdxList{idx(1)};
                        largest = false(size(pred));
                        largest(pixels) = true;
                        largest = imfill(largest, 'holes');
                        n_slice = size(largest,3);
                        if l >= 2
                            for n=1:n_slice
                                largest(:, :, n) = imfill(largest(:, :, n), 'holes');
                            end
                        end
                        clean_preds(largest) = l;
                    end
                else
                    continue;
                end
            end

            if strcmp(experiment.interp_type,'naive')
                clean_preds = uint16(clean_preds);
                clean_preds = imresize(clean_preds,...
                                                      [experiment.vol_size_orig(1), experiment.vol_size_orig(2)],...
                                                      'nearest');
            elseif  strcmp(experiment.interp_type,'same_fov')
                clean_preds = uint16(clean_preds);
                clean_preds = imresize(clean_preds,...
                                                      [experiment.vol_crop_size(1), experiment.vol_crop_size(2)],...
                                                      'nearest');
                clean_preds = permute(clean_preds, [1,3,2]);
                clean_preds = imresize(clean_preds, [experiment.vol_crop_size(1), experiment.vol_size_orig(3)], 'nearest');
                clean_preds = permute(clean_preds, [1,3,2]);
                clean_preds_rz(experiment.top_remove+1:end-experiment.bottom_remove,...
                                         experiment.left_remove+1:end-experiment.right_remove,...
                                         :) = clean_preds;
                clean_preds = clean_preds_rz;
            else
                error('Invalid interpolation type.')
            end
            
            experiment.processed_results = clean_preds;
        end
        
        function delete_images(experiment)
            delete(experiment.processed_image_path);
            delete(experiment.predictions_path);
        end
        
        function delete_configs(experiment)
            delete(experiment.configs_path);
        end
    end
end

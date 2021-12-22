%% Image spatial resolution
% preimplant
pre_spatial_res = [0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.31 0.31 1.0;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.43 0.43 1.0;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2;
                   0.33 0.33 1.2];
                          
pre_fov = [170   170
           170   170
           170   170
           170   170
           170   170
           170   170
           160   160
           170   170
           170   170
           170   170
           170   170
           170   170
           220   220
           170   170
           170   170
           170   170
           170   170
           170   170
           170   170
           170   170
           170   170
           170   170
           170   170
           170   170
           170   170];

% % postimplant
% post_spatial_res = [0.23 0.23 1.2;
%                     0.20 0.20 1.2;
%                     0.23 0.23 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.23 0.23 1.2;
%                     0.20 0.20 1.2;
%                     0.23 0.23 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.23 0.23 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.23 0.23 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.23 0.23 1.2;
%                     0.20 0.20 1.2;
%                     0.20 0.20 1.2;
%                     0.23 0.23 1.2];

%% Run experiments for all computer observers
encoders = {'UNet','VGG16','VGG19','DenseNet121','DenseNet169','DenseNet201','Xception','MobileNet','MobileNetV2','ResNet50','ResNet101','ResNet152','ResNet50V2','ResNet101V2','ResNet152V2','ResNeXt50','ResNeXt101','InceptionV3','InceptionResNetV2'};
losses = {'categorical_crossentropy','focal','jaccard','tversky'};
params = {{'0.3','0.7'},{'0.5','0.5'},{'0.7','0.30000000000000004'}};
models_dir = '/data/jsanders/iov_study/models/114_models_interp030312_white_noise_295pts';
images_dir = '/data/jsanders/iov_study/images';
mri_type = 'pre';
temp_dir = '/home/jsanders1-mda/github/dlae';
predictions_dir = '/data/jsanders/iov_study/predictions';
dlae_path = '/home/jsanders1-mda/github/dlae/dlae.py';
n_patients = 25;
config_template_fname = '/home/jsanders1-mda/github/dlae/configs/computational_observer_study_pelvic_mri_contour.json';
fid = fopen(config_template_fname);
configs = fread(fid,inf);
data_str = char(configs');
fclose(fid);
configs = jsondecode(data_str);

count=1;
for i=1:length(encoders)
    for j=1:length(losses)
        if strcmp(losses{j},'tversky')
            for k=1:length(param1s)
                if strcmp(param1s{k}, '0.5') && strcmp(param2s{k}, '0.5')
                    name_{count} = ['encoder_' encoders{i} '_loss_' losses{j} '_alpha_' param1s{k} '_beta_' param2s{k} '_ckpt.h5'];
                    count=count+1;
                end
            end
        else
            name_{count} = ['encoder_' encoders{i} '_loss_' losses{j} '_alpha_' param1s{1} '_beta_' param2s{1} '_ckpt.h5'];
            count=count+1;
        end
    end
end

experiments = [];
for i=1:length(name_)
    model_name = name_{i};
    model_path = [models_dir filesep model_name];
    configs.paths.load_model = model_path;
    
    for j=1:n_patients
        original_image_path = [images_dir filesep 'pt' num2str(j-1) filesep 'dlae' filesep 'pt' num2str(j-1) '.h5'];
        processed_image_path = [temp_dir filesep 'pt' num2str(j-1) '_processed.h5'];
        res_path = [images_dir filesep 'pt' num2str(j-1) filesep 'dlae' filesep 'pt' num2str(j-1) '_row_res.h5'];
        predictions_path = [predictions_dir filesep 'pt' num2str(j-1) '_processed_predictions.h5'];
        configs_path = [temp_dir filesep 'pt' num2str(j-1) '.json'];
        prediction_entry_path = dlae_path;
        
        experiment_queue(i,j).experiment =...
            ExperimentSeed(original_image_path, processed_image_path,...
                           res_path, predictions_path, configs_path,...
                           prediction_entry_path, 'mn3std_wn', 'same_fov', configs);
        experiments = [experiments, experiment_queue(i,j).experiment];
    end
end
    
for i=1:length(experiments)
    tic
    experiments(i) = experiments(i).run();
    toc
end

permanent_dir = '/data/jsanders/iov_study/raw_computer_predictions';
for i=1:length(experiments)
    tic
    fn_ = [permanent_dir filesep 'pt_' num2str(i-1) '.mat'];
    fn_preds_ = [permanent_dir filesep 'pt_' num2str(i-1) '_probability_map.mat'];
    image_ = experiments(i).original_images;
    preds_ = experiments(i).original_results;
    if size(preds_,1) ~= size(image_,1)
        sz_ = size(image_);
        preds_ = imresize(preds_,[sz_(1) sz_(2),'bicubic']);
    end
    if size(preds_,3) ~= size(image_,3)
        sz_ = size(image_);
        preds_ = permute(preds_,[1,3,2,4]);
        preds_ = imresize(preds_,[sz_(1),sz_(3)],'bicubic');
        preds_ = permute(preds_,[1,3,2,4]);
    end
    experiment_ = experiments(i);
    probability_map_ = preds_;
    save(fn_,'experiment_');
    save(fn_preds_,'probability_map_');
    toc
end

%% Figures of merit analysis
%reference observer from staple
ref_ = load('staple_pre_5obs_part2_03222021.mat');
ref_ = ref_.staple_;
thresh_ = 0.75;
for i=1:length(ref_)
    label_.contours(i).prostate = ref_(i).prostate>=thresh_;
    label_.contours(i).eus = ref_(i).eus>=thresh_;
    label_.contours(i).sv = ref_(i).sv>=thresh_;
    label_.contours(i).rectum = ref_(i).rectum>=thresh_;
    label_.contours(i).bladder = ref_(i).bladder>=thresh_;
end

% FCNeval predictions
n_patients = 25;
mri_type = 'pre';
model_ = 'DenseNet201';
loss_ = 'categorical_crossentropy_alpha_0.5_beta_0.5_ckpt';
permanent_dir = '/data/jsanders/iov_study/raw_computer_predictions';
for i=1:n_patients
    fn_exp_ = [permanent_dir filesep mri_type '_pt' num2str(i-1) '_encoder_' model_ '_loss_' loss_ '_probability_map.mat'];
    data_ = load(fn_exp_);
    exp_ = data_.probability_map_;
    probability_maps_(i).map = exp_;
end

thresh_ = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05,...
           0.1:0.0075:0.7, 0.75:0.0025:0.9,...
           0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,...
           0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999,...
           0.999999999, 0.9999999999, 0.99999999999, 0.999999999999, 0.9999999999999, 0.999999999999999];
n_thresh = length(thresh_);
for i=1:n_patients
    tp_p = []; fp_p = []; tn_p = []; fn_p = [];
    tp_e = []; fp_e = []; tn_e = []; fn_e = [];
    tp_s = []; fp_s = []; tn_s = []; fn_s = [];
    tp_r = []; fp_r = []; tn_r = []; fn_r = [];
    tp_b = []; fp_b = []; tn_b = []; fn_b = [];
    
    % account for the fact that the preimplant MRIs had variable FOVs
    if strcmp(mri_type,'pre')
        n_remove_r = (pre_fov(i,1) - 150) / pre_spatial_res(i,1);
        n_remove_c = (pre_fov(i,2) - 150) / pre_spatial_res(i,2);
        if n_remove_r < 0
            n_remove_r = 0;
        end
        if n_remove_c < 0
            n_remove_c = 0;
        end
        top_remove = ceil(n_remove_r/2);
        bottom_remove = ceil(n_remove_r/2);
        left_remove = ceil(n_remove_c/2);
        right_remove = ceil(n_remove_c/2);
        map_ = probability_maps_(i).map(top_remove+1:end-bottom_remove,...
                                        left_remove+1:end-right_remove,...
                                        :,:);
        mask_p = label_.contours(i).prostate(top_remove+1:end-bottom_remove,...
                                             left_remove+1:end-right_remove,...
                                             :);
        mask_e = label_.contours(i).eus(top_remove+1:end-bottom_remove,...
                                        left_remove+1:end-right_remove,...
                                        :);
        mask_s = label_.contours(i).sv(top_remove+1:end-bottom_remove,...
                                       left_remove+1:end-right_remove,...
                                       :);
        mask_r = label_.contours(i).rectum(top_remove+1:end-bottom_remove,...
                                           left_remove+1:end-right_remove,...
                                           :);
        mask_b = label_.contours(i).bladder(top_remove+1:end-bottom_remove,...
                                            left_remove+1:end-right_remove,...
                                            :);

        label_.contours(i).prostate = imresize(mask_p,[512,512],'nearest');
        label_.contours(i).eus = imresize(mask_e,[512,512],'nearest');
        label_.contours(i).sv = imresize(mask_s,[512,512],'nearest');
        label_.contours(i).rectum = imresize(mask_r,[512,512],'nearest');
        label_.contours(i).bladder = imresize(mask_b,[512,512],'nearest');
    end
    
    for j=1:length(thresh_)
        mask_p = probability_maps_(i).map(:,:,:,2)>=thresh_(j);
        mask_e = probability_maps_(i).map(:,:,:,3)>=thresh_(j);
        mask_s = probability_maps_(i).map(:,:,:,4)>=thresh_(j);
        mask_r = probability_maps_(i).map(:,:,:,5)>=thresh_(j);
        mask_b = probability_maps_(i).map(:,:,:,6)>=thresh_(j);
        
        % prostate
        tps = mask_p & label_.contours(i).prostate;
        tp_p(j) = sum(tps(:));
        fps = mask_p & ~label_.contours(i).prostate;
        fp_p(j) = sum(fps(:));
        tns = ~mask_p & ~label_.contours(i).prostate;
        tn_p(j) = sum(tns(:));
        fns = ~mask_p & label_.contours(i).prostate;
        fn_p(j) = sum(fns(:));
        
        % EUS
        tps = mask_e & label_.contours(i).eus;
        tp_e(j) = sum(tps(:));
        fps = mask_e & ~label_.contours(i).eus;
        fp_e(j) = sum(fps(:));
        tns = ~mask_e & ~label_.contours(i).eus;
        tn_e(j) = sum(tns(:));
        fns = ~mask_e & label_.contours(i).eus;
        fn_e(j) = sum(fns(:));
        
        % SV
        tps = mask_s & label_.contours(i).sv;
        tp_s(j) = sum(tps(:));
        fps = mask_s & ~label_.contours(i).sv;
        fp_s(j) = sum(fps(:));
        tns = ~mask_s & ~label_.contours(i).sv;
        tn_s(j) = sum(tns(:));
        fns = ~mask_s & label_.contours(i).sv;
        fn_s(j) = sum(fns(:));
        
        % rectum
        tps = mask_r & label_.contours(i).rectum;
        tp_r(j) = sum(tps(:));
        fps = mask_r & ~label_.contours(i).rectum;
        fp_r(j) = sum(fps(:));
        tns = ~mask_r & ~label_.contours(i).rectum;
        tn_r(j) = sum(tns(:));
        fns = ~mask_r & label_.contours(i).rectum;
        fn_r(j) = sum(fns(:));
        
        % bladder
        tps = mask_b & label_.contours(i).bladder;
        tp_b(j) = sum(tps(:));
        fps = mask_b & ~label_.contours(i).bladder;
        fp_b(j) = sum(fps(:));
        tns = ~mask_b & ~label_.contours(i).bladder;
        tn_b(j) = sum(tns(:));
        fns = ~mask_b & label_.contours(i).bladder;
        fn_b(j) = sum(fns(:));
    end
    
    precision_p = tp_p ./ (tp_p + fp_p);
    recall_p = tp_p ./ (tp_p + fn_p);
    f1_p = tp_p ./ (tp_p + mean([fp_p; fn_p]',2)');
    mcc_p = (tp_p.*tn_p - fp_p.*fn_p) ./ sqrt((tp_p + fp_p) .* (tp_p + fn_p) .* (tn_p + fp_p) .* (tn_p + fn_p));
    ji_p = tp_p ./ (tp_p + fp_p + fn_p);
    
    metrics_p(i).precision = precision_p;
    metrics_p(i).recall = recall_p;
    metrics_p(i).f1 = f1_p;
    metrics_p(i).mcc = mcc_p;
    metrics_p(i).ji = ji_p;
    
    precision_e = tp_e ./ (tp_e + fp_e);
    recall_e = tp_e ./ (tp_e + fn_e);
    f1_e = tp_e ./ (tp_e + mean([fp_e; fn_e]',2)');
    mcc_e = (tp_e.*tn_e - fp_e.*fn_e) ./ sqrt((tp_e + fp_e) .* (tp_e + fn_e) .* (tn_e + fp_e) .* (tn_e + fn_e));
    ji_e = tp_e ./ (tp_e + fp_e + fn_e);
    
    metrics_e(i).precision = precision_e;
    metrics_e(i).recall = recall_e;
    metrics_e(i).f1 = f1_e;
    metrics_e(i).mcc = mcc_e;
    metrics_e(i).ji = ji_e;
    
    precision_s = tp_s ./ (tp_s + fp_s);
    recall_s = tp_s ./ (tp_s + fn_s);
    f1_s = tp_s ./ (tp_s + mean([fp_s; fn_s]',2)');
    mcc_s = (tp_s.*tn_s - fp_s.*fn_s) ./ sqrt((tp_s + fp_s) .* (tp_s + fn_s) .* (tn_s + fp_s) .* (tn_s + fn_s));
    ji_s = tp_s ./ (tp_s + fp_s + fn_s);
    
    metrics_s(i).precision = precision_s;
    metrics_s(i).recall = recall_s;
    metrics_s(i).f1 = f1_s;
    metrics_s(i).mcc = mcc_s;
    metrics_s(i).ji = ji_s;
    
    precision_r = tp_r ./ (tp_r + fp_r);
    recall_r = tp_r ./ (tp_r + fn_r);
    f1_r = tp_r ./ (tp_r + mean([fp_r; fn_r]',2)');
    mcc_r = (tp_r.*tn_r - fp_r.*fn_r) ./ sqrt((tp_r + fp_r) .* (tp_r + fn_r) .* (tn_r + fp_r) .* (tn_r + fn_r));
    ji_r = tp_r ./ (tp_r + fp_r + fn_r);
    
    metrics_r(i).precision = precision_r;
    metrics_r(i).recall = recall_r;
    metrics_r(i).f1 = f1_r;
    metrics_r(i).mcc = mcc_r;
    metrics_r(i).ji = ji_r;
    
    precision_b = tp_b ./ (tp_b + fp_b);
    recall_b = tp_b ./ (tp_b + fn_b);
    f1_b = tp_b ./ (tp_b + mean([fp_b; fn_b]',2)');
    mcc_b = (tp_b.*tn_b - fp_b.*fn_b) ./ sqrt((tp_b + fp_b) .* (tp_b + fn_b) .* (tn_b + fp_b) .* (tn_b + fn_b));
    ji_b = tp_b ./ (tp_b + fp_b + fn_b);
    
    metrics_b(i).precision = precision_b;
    metrics_b(i).recall = recall_b;
    metrics_b(i).f1 = f1_b;
    metrics_b(i).mcc = mcc_b;
    metrics_b(i).ji = ji_b;
end
experiment_results.prostate = metrics_p;
experiment_results.eus = metrics_e;
experiment_results.sv = metrics_s;
experiment_results.rectum = metrics_r;
experiment_results.bladder = metrics_b;
% save(['experiment_results_' num2str(n_patients) '_' mri_type '.mat'],'experiment_results')

%% Spatial entropy mapping
encoders_ = {'UNet','VGG16','VGG19','DenseNet121','DenseNet169','DenseNet201','Xception','MobileNet','MobileNetV2','ResNet50','ResNet101','ResNet152','ResNet50V2','ResNet101V2','ResNet152V2','ResNeXt50','ResNeXt101','InceptionV3','InceptionResNetV2'};
losses_ = {'categorical_crossentropy_alpha_0.5_beta_0.5','focal_alpha_0.5_beta_0.5','jaccard_alpha_0.5_beta_0.5','tversky_alpha_0.3_beta_0.7','tversky_alpha_0.5_beta_0.5','tversky_alpha_0.7_beta_0.30000000000000004'};
losses_ = {'tversky_alpha_0.5_beta_0.5'};
permanent_dir = '/data/jsanders/iov_study/raw_computer_predictions';
count=1;
for i=1:length(encoders_)
    for j=1:length(losses_)
        name_{count} = ['encoder_' encoders_{i} '_loss_' losses_{j} '_ckpt'];
        count=count+1;
    end
end

n_patients = 25;
n_models = 114;
inds_ = 1:n_models;
mri_type = 'pre';

% at argmax
for i=1:n_patients
    for j=1:length(inds_)
        fn_exp_ = [permanent_dir filesep mri_type '_pt' num2str(i-1) '_' name_{inds_(j)} '_probability_map.mat'];
        data_ = load(fn_exp_);
        exp_ = data_.probability_map_;
        [~,argmax_] = max(exp_,[],4);
        exp_ = uint8(argmax_-1);
        patient_(i).masks(j).argmax_mask = exp_;
    end
end

% at fixed operating point
thresh_p = 0.31;
% thresh_p = 0.3175;
thresh_e = 0.05;
% thresh_e = 0.2050;
thresh_s = 0.2275;
% thresh_s = 0.3400;
thresh_r = 0.445;
% thresh_r = 0.4225;
thresh_b = 0.37;
% thresh_b = 0.4000;
for i=1:n_patients
    masks = [];
    parfor j=1:length(inds_)
        fn_exp_ = [permanent_dir filesep mri_type '_pt' num2str(i-1) '_' name_{inds_(j)} '_probability_map.mat'];
        data_ = load(fn_exp_);
        exp_ = data_.probability_map_;
        sz_ = size(exp_);
        mask_ = zeros(sz_(1:3));
        p_ = exp_(:,:,:,2)>=thresh_p;
        e_ = exp_(:,:,:,3)>=thresh_e;
        s_ = exp_(:,:,:,4)>=thresh_s;
        r_ = exp_(:,:,:,5)>=thresh_r;
        b_ = exp_(:,:,:,6)>=thresh_b;
        mask_(b_) = 5;
        mask_(r_) = 4;
        mask_(s_) = 3;
        mask_(p_) = 1;
        mask_(e_) = 2;
        mask_ = uint8(mask_);
        masks(j).thresh_mask = mask_;
    end
    patient_(i).masks = masks;
end

%map all models
for i=1:length(patient_)
    sz = size(patient_(i).masks(1).thresh_mask);
    entropy_to_map(i).voxel_data = uint8(zeros([sz, n_models]));
    for j=1:length(patient_(i).masks)
        entropy_to_map(i).voxel_data(:,:,:,j) = patient_(i).masks(j).argmax_mask;
%         entropy_to_map(i).voxel_data(:,:,:,j) = patient_(i).masks(j).thresh_mask;
    end
end

n_organs = 5;
for i=1:length(entropy_to_map)
    voxel_data_ = entropy_to_map(i).voxel_data;
    sz = size(voxel_data_);
    sum_ = sum(voxel_data_,4);
    idx = find(sum_ > 0);
    H_ = zeros(length(idx),1);
    entropy_mask = zeros(sz(1:3));
    raw_measurement_mask = voxel_data_;

    parfor j=1:length(idx)
        [r_,c_,s_] = ind2sub([sz(1),sz(2),sz(3)],idx(j));
        data = double(squeeze(raw_measurement_mask(r_,c_,s_,:)));
        [counts,cents] = hist(data,n_organs+1);
        n_ = counts>0;
        frequency = counts(n_)/sum(counts(n_));
        H_(j) = -sum(frequency.*log2(frequency));
    end
    entropy_mask(idx) = H_;
    spatial_entropy_maps(i).spatial_entropy_map = entropy_mask;
end
% save('entropy_maps_post_all_114_models_03192021.mat','-v7.3','spatial_entropy_maps')

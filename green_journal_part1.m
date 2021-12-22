%% Image spatial resolution
% voxel sizes for pre-implant MRIs
spatial_res = [0.33 0.33 1.2;
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

% % voxel sizes for post-implant MRIs
% spatial_res = [0.23 0.23 1.2;
%                0.20 0.20 1.2;
%                0.23 0.23 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.23 0.23 1.2;
%                0.20 0.20 1.2;
%                0.23 0.23 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.23 0.23 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.23 0.23 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;
%                0.23 0.23 1.2;
%                0.20 0.20 1.2;
%                0.20 0.20 1.2;            
%                0.23 0.23 1.2];

%% Observer contours
% load observer contours
o1 = load('contours/contours_js_pre.mat');
o2 = load('contours/contours_sf_pre.mat');
o3 = load('contours/contours_hm_pre.mat');
o4 = load('contours/contours_ah_pre.mat');
o5 = load('contours/contours_ct_pre.mat');
o6 = load('contours/contours_av_pre.mat');
o7 = load('contours/contours_tlb_pre.mat');
% o1 = load('contours/contours_js_post.mat');
% o2 = load('contours/contours_sf_post.mat');
% o3 = load('contours/contours_hm_post.mat');
% o4 = load('contours/contours_ah_post.mat');
% o5 = load('contours/contours_ct_post.mat');
% o6 = load('contours/contours_av_post.mat');
% o7 = load('contours/contours_tlb_post.mat');

%% Human observers
% create the observers
o1 = ContouringObserver(o1.contours, spatial_res);
o2 = ContouringObserver(o2.contours, spatial_res);
o3 = ContouringObserver(o3.contours, spatial_res);
o4 = ContouringObserver(o4.contours, spatial_res);
o5 = ContouringObserver(o5.contours, spatial_res);
o6 = ContouringObserver(o6.contours, spatial_res);
o7 = ContouringObserver(o7.contours, spatial_res);
human_observers = [o1, o2, o3, o4, o5, o6, o7];

% merge all observer masks into one array
for i=1:length(human_observers)
    for j=1:length(human_observers(i).contours)
        human_observers(i).contours(j).organ_mask = zeros(size(human_observers(i).contours(j).prostate));
        human_observers(i).contours(j).organ_mask(logical(human_observers(i).contours(j).sv)) = 3;
        human_observers(i).contours(j).organ_mask(logical(human_observers(i).contours(j).rectum)) = 4;
        human_observers(i).contours(j).organ_mask(logical(human_observers(i).contours(j).bladder)) = 5;
        human_observers(i).contours(j).organ_mask(logical(human_observers(i).contours(j).prostate)) = 1;
        human_observers(i).contours(j).organ_mask(logical(human_observers(i).contours(j).eus)) = 2;
    end
end

% organ-wise probability maps
h_joint_observations = combine_observations(human_observers);

%% o1
% o1 as the reference observer. compare all human observers
o1_o1_metrics = o1.compare(o1);
o1_o2_metrics = o1.compare(o2);
o1_o3_metrics = o1.compare(o3);
o1_o4_metrics = o1.compare(o4);
o1_o5_metrics = o1.compare(o5);
o1_o6_metrics = o1.compare(o6);
o1_o7_metrics = o1.compare(o7);

% human no contest comparison metrics. no computer mask predictions
% contained in this population
o1_nc_metrics = o1.compare(h_no_contest_observer);
o1_metrics = [o1_o1_metrics o1_o2_metrics o1_o3_metrics o1_o4_metrics o1_o5_metrics o1_o6_metrics o1_o7_metrics];
o1_metrics = Metrics(o1_metrics);

%% o2
% o2 as the reference observer. compare all human observers
o2_o1_metrics = o2.compare(o1);
o2_o2_metrics = o2.compare(o2);
o2_o3_metrics = o2.compare(o3);
o2_o4_metrics = o2.compare(o4);
o2_o5_metrics = o2.compare(o5);
o2_o6_metrics = o2.compare(o6);
o2_o7_metrics = o2.compare(o7);

%% o3
% o3 as the reference observer. compare all human observers
o3_o1_metrics = o3.compare(o1);
o3_o2_metrics = o3.compare(o2);
o3_o3_metrics = o3.compare(o3);
o3_o4_metrics = o3.compare(o4);
o3_o5_metrics = o3.compare(o5);
o3_o6_metrics = o3.compare(o6);
o3_o7_metrics = o3.compare(o7);

%% o4
% o4 as the reference observer. compare all human observers
o4_o1_metrics = o4.compare(o1);
o4_o2_metrics = o4.compare(o2);
o4_o3_metrics = o4.compare(o3);
o4_o4_metrics = o4.compare(o4);
o4_o5_metrics = o4.compare(o5);
o4_o6_metrics = o4.compare(o6);
o4_o7_metrics = o4.compare(o7);

%% o5
% o5 as the reference observer. compare all human observers
o5_o1_metrics = o5.compare(o1);
o5_o2_metrics = o5.compare(o2);
o5_o3_metrics = o5.compare(o3);
o5_o4_metrics = o5.compare(o4);
o5_o5_metrics = o5.compare(o5);
o5_o6_metrics = o5.compare(o6);
o5_o7_metrics = o5.compare(o7);

%% o6
% computer as the reference observer. compare all human observers
o6_o1_metrics = o6.compare(o1);
o6_o2_metrics = o6.compare(o2);
o6_o3_metrics = o6.compare(o3);
o6_o4_metrics = o6.compare(o4);
o6_o5_metrics = o6.compare(o5);
o6_o6_metrics = o6.compare(o6);
o6_o7_metrics = o6.compare(o7);

%% o7
% computer as the reference observer. compare all human observers
o7_o1_metrics = o7.compare(o1);
o7_o2_metrics = o7.compare(o2);
o7_o3_metrics = o7.compare(o3);
o7_o4_metrics = o7.compare(o4);
o7_o5_metrics = o7.compare(o5);
o7_o6_metrics = o7.compare(o6);
o7_o7_metrics = o7.compare(o7);

%% Store data

% save everything
save('iov_study_data_preimplant_01032021_7humans.mat','-v7.3',...
     'o1','o2','o3','o4','o5','o6','o7',...
     'o1_o1_metrics','o1_o2_metrics','o1_o3_metrics','o1_o4_metrics','o1_o5_metrics','o1_o6_metrics','o1_o7_metrics',...
     'o2_o1_metrics','o2_o2_metrics','o2_o3_metrics','o2_o4_metrics','o2_o5_metrics','o2_o6_metrics','o2_o7_metrics',...
     'o3_o1_metrics','o3_o2_metrics','o3_o3_metrics','o3_o4_metrics','o3_o5_metrics','o3_o6_metrics','o3_o7_metrics',...
     'o4_o1_metrics','o4_o2_metrics','o4_o3_metrics','o4_o4_metrics','o4_o5_metrics','o4_o6_metrics','o4_o7_metrics',...
     'o5_o1_metrics','o5_o2_metrics','o5_o3_metrics','o5_o4_metrics','o5_o5_metrics','o5_o6_metrics','o5_o7_metrics',...
     'o6_o1_metrics','o6_o2_metrics','o6_o3_metrics','o6_o4_metrics','o6_o5_metrics','o6_o6_metrics','o6_o7_metrics',...
     'o1_metrics','o2_metrics','o3_metrics','o4_metrics','o5_metrics','o6_metrics','o7_metrics',...
     'h_joint_observations','spatial_res','human_observers')

%% Compare human observers
% human comparisons
all_comparisons = [o1_o2_metrics o1_o3_metrics o1_o4_metrics o1_o5_metrics o1_o6_metrics o1_o7_metrics,...
                                 o2_o3_metrics o2_o4_metrics o2_o5_metrics o2_o6_metrics o2_o7_metrics,...
                                               o3_o4_metrics o3_o5_metrics o3_o6_metrics o3_o7_metrics,...
                                                             o4_o5_metrics o4_o6_metrics o4_o7_metrics,...
                                                                           o5_o6_metrics o5_o7_metrics,...
                                                                                         o6_o7_metrics];

% all unique comparisons amongst ROs
all_ro_comparisons = [o2_o3_metrics o2_o5_metrics o3_o5_metrics];

% all unique comparisons amongst least frequent contourers
all_lfc_comparisons = [o1_o4_metrics o1_o6_metrics o1_o7_metrics...
                                     o4_o6_metrics o4_o7_metrics...
                                                   o6_o7_metrics];

% all unique comparisons for 5 observers in part 2
all_comparisons_part2 = [o1_o2_metrics o1_o4_metrics o1_o5_metrics o1_o7_metrics,...
                                       o2_o4_metrics o2_o5_metrics o2_o7_metrics,...
                                                     o4_o5_metrics o4_o7_metrics,...
                                                                   o5_o7_metrics];

%% Save observations
comparisons_ = all_comparisons;
metrics_ = Metrics(comparisons_).combine_metrics();

% prostate
p_ = [metrics_.grouped_metrics.prostate.precision'...
      metrics_.grouped_metrics.prostate.recall'...
      metrics_.grouped_metrics.prostate.dsc'...
      metrics_.grouped_metrics.prostate.mcc'...
      metrics_.grouped_metrics.prostate.jaccard'];

% EUS
e_ = [metrics_.grouped_metrics.eus.precision'...
      metrics_.grouped_metrics.eus.recall'...
      metrics_.grouped_metrics.eus.dsc'...
      metrics_.grouped_metrics.eus.mcc'...
      metrics_.grouped_metrics.eus.jaccard'];

% SV
s_ = [metrics_.grouped_metrics.sv.precision'...
      metrics_.grouped_metrics.sv.recall'...
      metrics_.grouped_metrics.sv.dsc'...
      metrics_.grouped_metrics.sv.mcc'...
      metrics_.grouped_metrics.sv.jaccard'];

% rectum
r_ = [metrics_.grouped_metrics.rectum.precision'...
      metrics_.grouped_metrics.rectum.recall'...
      metrics_.grouped_metrics.rectum.dsc'...
      metrics_.grouped_metrics.rectum.mcc'...
      metrics_.grouped_metrics.rectum.jaccard'];

% bladder
b_ = [metrics_.grouped_metrics.bladder.precision'...
      metrics_.grouped_metrics.bladder.recall'...
      metrics_.grouped_metrics.bladder.dsc'...
      metrics_.grouped_metrics.bladder.mcc'...
      metrics_.grouped_metrics.bladder.jaccard'];

xlswrite('preimplant_similarity_metric_summaries_green_journal_part1_7obs.xls', [p_ e_ s_ r_ b_]);

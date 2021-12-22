classdef Metrics
    properties
        observer_comparisons = []
        grouped_metrics = []
    end
    
    methods
        function metrics = Metrics(observer_metrics)
            for i=1:length(observer_metrics)
                metrics.observer_comparisons(i).prostate = observer_metrics(i).prostate;
                metrics.observer_comparisons(i).eus = observer_metrics(i).eus;
                metrics.observer_comparisons(i).sv = observer_metrics(i).sv;
                metrics.observer_comparisons(i).rectum = observer_metrics(i).rectum;
                metrics.observer_comparisons(i).bladder = observer_metrics(i).bladder;
            end
            
            metrics = metrics.combine_metrics();
        end
    end
    
    methods
        function metrics = combine_metrics(metrics)
            metrics.grouped_metrics.prostate.mcc = [];
            metrics.grouped_metrics.prostate.dsc = [];
            metrics.grouped_metrics.prostate.precision = [];
            metrics.grouped_metrics.prostate.recall = [];
            metrics.grouped_metrics.prostate.absolute_volume_difference = [];
            metrics.grouped_metrics.prostate.jaccard = [];
            metrics.grouped_metrics.prostate.pixel_accuracy = [];
            metrics.grouped_metrics.prostate.ppv = [];
            metrics.grouped_metrics.prostate.tps = [];
            metrics.grouped_metrics.prostate.fps = [];
            metrics.grouped_metrics.prostate.tns = [];
            metrics.grouped_metrics.prostate.fns = [];
            metrics.grouped_metrics.prostate.reference_vol = [];
            metrics.grouped_metrics.prostate.test_vol = [];
            
            metrics.grouped_metrics.eus.mcc = [];
            metrics.grouped_metrics.eus.dsc = [];
            metrics.grouped_metrics.eus.precision = [];
            metrics.grouped_metrics.eus.recall = [];
            metrics.grouped_metrics.eus.absolute_volume_difference = [];
            metrics.grouped_metrics.eus.jaccard = [];
            metrics.grouped_metrics.eus.pixel_accuracy = [];
            metrics.grouped_metrics.eus.ppv = [];
            metrics.grouped_metrics.eus.tps = [];
            metrics.grouped_metrics.eus.fps = [];
            metrics.grouped_metrics.eus.tns = [];
            metrics.grouped_metrics.eus.fns = [];
            metrics.grouped_metrics.eus.reference_vol = [];
            metrics.grouped_metrics.eus.test_vol = [];
            
            metrics.grouped_metrics.sv.mcc = [];
            metrics.grouped_metrics.sv.dsc = [];
            metrics.grouped_metrics.sv.precision = [];
            metrics.grouped_metrics.sv.recall = [];
            metrics.grouped_metrics.sv.absolute_volume_difference = [];
            metrics.grouped_metrics.sv.jaccard = [];
            metrics.grouped_metrics.sv.pixel_accuracy = [];
            metrics.grouped_metrics.sv.ppv = [];
            metrics.grouped_metrics.sv.tps = [];
            metrics.grouped_metrics.sv.fps = [];
            metrics.grouped_metrics.sv.tns = [];
            metrics.grouped_metrics.sv.fns = [];
            metrics.grouped_metrics.sv.reference_vol = [];
            metrics.grouped_metrics.sv.test_vol = [];
            
            metrics.grouped_metrics.rectum.mcc = [];
            metrics.grouped_metrics.rectum.dsc = [];
            metrics.grouped_metrics.rectum.precision = [];
            metrics.grouped_metrics.rectum.recall = [];
            metrics.grouped_metrics.rectum.absolute_volume_difference = [];
            metrics.grouped_metrics.rectum.jaccard = [];
            metrics.grouped_metrics.rectum.pixel_accuracy = [];
            metrics.grouped_metrics.rectum.ppv = [];
            metrics.grouped_metrics.rectum.tps = [];
            metrics.grouped_metrics.rectum.fps = [];
            metrics.grouped_metrics.rectum.tns = [];
            metrics.grouped_metrics.rectum.fns = [];
            metrics.grouped_metrics.rectum.reference_vol = [];
            metrics.grouped_metrics.rectum.test_vol = [];
            
            metrics.grouped_metrics.bladder.mcc = [];
            metrics.grouped_metrics.bladder.dsc = [];
            metrics.grouped_metrics.bladder.precision = [];
            metrics.grouped_metrics.bladder.recall = [];
            metrics.grouped_metrics.bladder.absolute_volume_difference = [];
            metrics.grouped_metrics.bladder.jaccard = [];
            metrics.grouped_metrics.bladder.pixel_accuracy = [];
            metrics.grouped_metrics.bladder.ppv = [];
            metrics.grouped_metrics.bladder.tps = [];
            metrics.grouped_metrics.bladder.fps = [];
            metrics.grouped_metrics.bladder.tns = [];
            metrics.grouped_metrics.bladder.fns = [];
            metrics.grouped_metrics.bladder.reference_vol = [];
            metrics.grouped_metrics.bladder.test_vol = [];
            
            for i=1:length(metrics.observer_comparisons)
                metrics.grouped_metrics.prostate.mcc = [metrics.grouped_metrics.prostate.mcc metrics.observer_comparisons(i).prostate.mcc];
                metrics.grouped_metrics.prostate.dsc = [metrics.grouped_metrics.prostate.dsc metrics.observer_comparisons(i).prostate.dsc];
                metrics.grouped_metrics.prostate.precision = [metrics.grouped_metrics.prostate.precision metrics.observer_comparisons(i).prostate.precision];
                metrics.grouped_metrics.prostate.recall = [metrics.grouped_metrics.prostate.recall metrics.observer_comparisons(i).prostate.recall];
                metrics.grouped_metrics.prostate.absolute_volume_difference = [metrics.grouped_metrics.prostate.absolute_volume_difference metrics.observer_comparisons(i).prostate.absolute_volume_difference];
                metrics.grouped_metrics.prostate.jaccard = [metrics.grouped_metrics.prostate.jaccard metrics.observer_comparisons(i).prostate.jaccard];
                metrics.grouped_metrics.prostate.pixel_accuracy = [metrics.grouped_metrics.prostate.pixel_accuracy metrics.observer_comparisons(i).prostate.pixel_accuracy];
                metrics.grouped_metrics.prostate.ppv = [metrics.grouped_metrics.prostate.ppv metrics.observer_comparisons(i).prostate.ppv];
                metrics.grouped_metrics.prostate.tps = [metrics.grouped_metrics.prostate.tps metrics.observer_comparisons(i).prostate.tps];
                metrics.grouped_metrics.prostate.fps = [metrics.grouped_metrics.prostate.fps metrics.observer_comparisons(i).prostate.fps];
                metrics.grouped_metrics.prostate.tns = [metrics.grouped_metrics.prostate.tns metrics.observer_comparisons(i).prostate.tns];
                metrics.grouped_metrics.prostate.fns = [metrics.grouped_metrics.prostate.fns metrics.observer_comparisons(i).prostate.fns];
                metrics.grouped_metrics.prostate.reference_vol = [metrics.grouped_metrics.prostate.reference_vol metrics.observer_comparisons(i).prostate.reference_vol];
                metrics.grouped_metrics.prostate.test_vol = [metrics.grouped_metrics.prostate.test_vol metrics.observer_comparisons(i).prostate.test_vol];

                metrics.grouped_metrics.eus.mcc = [metrics.grouped_metrics.eus.mcc metrics.observer_comparisons(i).eus.mcc];
                metrics.grouped_metrics.eus.dsc = [metrics.grouped_metrics.eus.dsc metrics.observer_comparisons(i).eus.dsc];
                metrics.grouped_metrics.eus.precision = [metrics.grouped_metrics.eus.precision metrics.observer_comparisons(i).eus.precision];
                metrics.grouped_metrics.eus.recall = [metrics.grouped_metrics.eus.recall metrics.observer_comparisons(i).eus.recall];
                metrics.grouped_metrics.eus.absolute_volume_difference = [metrics.grouped_metrics.eus.absolute_volume_difference metrics.observer_comparisons(i).eus.absolute_volume_difference];
                metrics.grouped_metrics.eus.jaccard = [metrics.grouped_metrics.eus.jaccard metrics.observer_comparisons(i).eus.jaccard];
                metrics.grouped_metrics.eus.pixel_accuracy = [metrics.grouped_metrics.eus.pixel_accuracy metrics.observer_comparisons(i).eus.pixel_accuracy];
                metrics.grouped_metrics.eus.ppv = [metrics.grouped_metrics.eus.ppv metrics.observer_comparisons(i).eus.ppv];
                metrics.grouped_metrics.eus.tps = [metrics.grouped_metrics.eus.tps metrics.observer_comparisons(i).eus.tps];
                metrics.grouped_metrics.eus.fps = [metrics.grouped_metrics.eus.fps metrics.observer_comparisons(i).eus.fps];
                metrics.grouped_metrics.eus.tns = [metrics.grouped_metrics.eus.tns metrics.observer_comparisons(i).eus.tns];
                metrics.grouped_metrics.eus.fns = [metrics.grouped_metrics.eus.fns metrics.observer_comparisons(i).eus.fns];
                metrics.grouped_metrics.eus.reference_vol = [metrics.grouped_metrics.eus.reference_vol metrics.observer_comparisons(i).eus.reference_vol];
                metrics.grouped_metrics.eus.test_vol = [metrics.grouped_metrics.eus.test_vol metrics.observer_comparisons(i).eus.test_vol];
                
                metrics.grouped_metrics.sv.mcc = [metrics.grouped_metrics.sv.mcc metrics.observer_comparisons(i).sv.mcc];
                metrics.grouped_metrics.sv.dsc = [metrics.grouped_metrics.sv.dsc metrics.observer_comparisons(i).sv.dsc];
                metrics.grouped_metrics.sv.precision = [metrics.grouped_metrics.sv.precision metrics.observer_comparisons(i).sv.precision];
                metrics.grouped_metrics.sv.recall = [metrics.grouped_metrics.sv.recall metrics.observer_comparisons(i).sv.recall];
                metrics.grouped_metrics.sv.absolute_volume_difference = [metrics.grouped_metrics.sv.absolute_volume_difference metrics.observer_comparisons(i).sv.absolute_volume_difference];
                metrics.grouped_metrics.sv.jaccard = [metrics.grouped_metrics.sv.jaccard metrics.observer_comparisons(i).sv.jaccard];
                metrics.grouped_metrics.sv.pixel_accuracy = [metrics.grouped_metrics.sv.pixel_accuracy metrics.observer_comparisons(i).sv.pixel_accuracy];
                metrics.grouped_metrics.sv.ppv = [metrics.grouped_metrics.sv.ppv metrics.observer_comparisons(i).sv.ppv];
                metrics.grouped_metrics.sv.tps = [metrics.grouped_metrics.sv.tps metrics.observer_comparisons(i).sv.tps];
                metrics.grouped_metrics.sv.fps = [metrics.grouped_metrics.sv.fps metrics.observer_comparisons(i).sv.fps];
                metrics.grouped_metrics.sv.tns = [metrics.grouped_metrics.sv.tns metrics.observer_comparisons(i).sv.tns];
                metrics.grouped_metrics.sv.fns = [metrics.grouped_metrics.sv.fns metrics.observer_comparisons(i).sv.fns];
                metrics.grouped_metrics.sv.reference_vol = [metrics.grouped_metrics.sv.reference_vol metrics.observer_comparisons(i).sv.reference_vol];
                metrics.grouped_metrics.sv.test_vol = [metrics.grouped_metrics.sv.test_vol metrics.observer_comparisons(i).sv.test_vol];
                
                metrics.grouped_metrics.rectum.mcc = [metrics.grouped_metrics.rectum.mcc metrics.observer_comparisons(i).rectum.mcc];
                metrics.grouped_metrics.rectum.dsc = [metrics.grouped_metrics.rectum.dsc metrics.observer_comparisons(i).rectum.dsc];
                metrics.grouped_metrics.rectum.precision = [metrics.grouped_metrics.rectum.precision metrics.observer_comparisons(i).rectum.precision];
                metrics.grouped_metrics.rectum.recall = [metrics.grouped_metrics.rectum.recall metrics.observer_comparisons(i).rectum.recall];
                metrics.grouped_metrics.rectum.absolute_volume_difference = [metrics.grouped_metrics.rectum.absolute_volume_difference metrics.observer_comparisons(i).rectum.absolute_volume_difference];
                metrics.grouped_metrics.rectum.jaccard = [metrics.grouped_metrics.rectum.jaccard metrics.observer_comparisons(i).rectum.jaccard];
                metrics.grouped_metrics.rectum.pixel_accuracy = [metrics.grouped_metrics.rectum.pixel_accuracy metrics.observer_comparisons(i).rectum.pixel_accuracy];
                metrics.grouped_metrics.rectum.ppv = [metrics.grouped_metrics.rectum.ppv metrics.observer_comparisons(i).rectum.ppv];
                metrics.grouped_metrics.rectum.tps = [metrics.grouped_metrics.rectum.tps metrics.observer_comparisons(i).rectum.tps];
                metrics.grouped_metrics.rectum.fps = [metrics.grouped_metrics.rectum.fps metrics.observer_comparisons(i).rectum.fps];
                metrics.grouped_metrics.rectum.tns = [metrics.grouped_metrics.rectum.tns metrics.observer_comparisons(i).rectum.tns];
                metrics.grouped_metrics.rectum.fns = [metrics.grouped_metrics.rectum.fns metrics.observer_comparisons(i).rectum.fns];
                metrics.grouped_metrics.rectum.reference_vol = [metrics.grouped_metrics.rectum.reference_vol metrics.observer_comparisons(i).rectum.reference_vol];
                metrics.grouped_metrics.rectum.test_vol = [metrics.grouped_metrics.rectum.test_vol metrics.observer_comparisons(i).rectum.test_vol];
                
                metrics.grouped_metrics.bladder.mcc = [metrics.grouped_metrics.bladder.mcc metrics.observer_comparisons(i).bladder.mcc];
                metrics.grouped_metrics.bladder.dsc = [metrics.grouped_metrics.bladder.dsc metrics.observer_comparisons(i).bladder.dsc];
                metrics.grouped_metrics.bladder.precision = [metrics.grouped_metrics.bladder.precision metrics.observer_comparisons(i).bladder.precision];
                metrics.grouped_metrics.bladder.recall = [metrics.grouped_metrics.bladder.recall metrics.observer_comparisons(i).bladder.recall];
                metrics.grouped_metrics.bladder.absolute_volume_difference = [metrics.grouped_metrics.bladder.absolute_volume_difference metrics.observer_comparisons(i).bladder.absolute_volume_difference];
                metrics.grouped_metrics.bladder.jaccard = [metrics.grouped_metrics.bladder.jaccard metrics.observer_comparisons(i).bladder.jaccard];
                metrics.grouped_metrics.bladder.pixel_accuracy = [metrics.grouped_metrics.bladder.pixel_accuracy metrics.observer_comparisons(i).bladder.pixel_accuracy];
                metrics.grouped_metrics.bladder.ppv = [metrics.grouped_metrics.bladder.ppv metrics.observer_comparisons(i).bladder.ppv];
                metrics.grouped_metrics.bladder.tps = [metrics.grouped_metrics.bladder.tps metrics.observer_comparisons(i).bladder.tps];
                metrics.grouped_metrics.bladder.fps = [metrics.grouped_metrics.bladder.fps metrics.observer_comparisons(i).bladder.fps];
                metrics.grouped_metrics.bladder.tns = [metrics.grouped_metrics.bladder.tns metrics.observer_comparisons(i).bladder.tns];
                metrics.grouped_metrics.bladder.fns = [metrics.grouped_metrics.bladder.fns metrics.observer_comparisons(i).bladder.fns];
                metrics.grouped_metrics.bladder.reference_vol = [metrics.grouped_metrics.bladder.reference_vol metrics.observer_comparisons(i).bladder.reference_vol];
                metrics.grouped_metrics.bladder.test_vol = [metrics.grouped_metrics.bladder.test_vol metrics.observer_comparisons(i).bladder.test_vol];
            end
        end
    end
end
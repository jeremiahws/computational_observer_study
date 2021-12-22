classdef ContouringObserver
    
    properties
        contours
        image_resolution
    end
    
    methods
        function observer = ContouringObserver(contours, res)
            observer.contours = contours;
            observer.image_resolution = res;
        end
        
        function metrics = compare(observer, observer2)
            for i=1:length(observer.contours)
                metrics.prostate.mcc(i) = observer.matthews_corr_coef(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.mcc(i) = observer.matthews_corr_coef(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.mcc(i) = observer.matthews_corr_coef(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.mcc(i) = observer.matthews_corr_coef(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.mcc(i) = observer.matthews_corr_coef(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.dsc(i) = observer.dice(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.dsc(i) = observer.dice(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.dsc(i) = observer.dice(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.dsc(i) = observer.dice(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.dsc(i) = observer.dice(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.precision(i) = observer.precision(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.precision(i) = observer.precision(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.precision(i) = observer.precision(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.precision(i) = observer.precision(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.precision(i) = observer.precision(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.recall(i) = observer.recall(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.recall(i) = observer.recall(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.recall(i) = observer.recall(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.recall(i) = observer.recall(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.recall(i) = observer.recall(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.absolute_volume_difference(i) = observer.absolute_volume_difference(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.absolute_volume_difference(i) = observer.absolute_volume_difference(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.absolute_volume_difference(i) = observer.absolute_volume_difference(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.absolute_volume_difference(i) = observer.absolute_volume_difference(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.absolute_volume_difference(i) = observer.absolute_volume_difference(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.jaccard(i) = observer.jaccard(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.jaccard(i) = observer.jaccard(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.jaccard(i) = observer.jaccard(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.jaccard(i) = observer.jaccard(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.jaccard(i) = observer.jaccard(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.tps(i) = observer.tps_(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.tps(i) = observer.tps_(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.tps(i) = observer.tps_(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.tps(i) = observer.tps_(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.tps(i) = observer.tps_(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.fps(i) = observer.fps_(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.fps(i) = observer.fps_(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.fps(i) = observer.fps_(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.fps(i) = observer.fps_(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.fps(i) = observer.fps_(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.tns(i) = observer.tns_(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.tns(i) = observer.tns_(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.tns(i) = observer.tns_(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.tns(i) = observer.tns_(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.tns(i) = observer.tns_(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.fns(i) = observer.fns_(observer.contours(i).prostate, observer2.contours(i).prostate);
                metrics.eus.fns(i) = observer.fns_(observer.contours(i).eus, observer2.contours(i).eus);
                metrics.sv.fns(i) = observer.fns_(observer.contours(i).sv, observer2.contours(i).sv);
                metrics.rectum.fns(i) = observer.fns_(observer.contours(i).rectum, observer2.contours(i).rectum);
                metrics.bladder.fns(i) = observer.fns_(observer.contours(i).bladder, observer2.contours(i).bladder);
                
                metrics.prostate.reference_vol(i) = observer.volume(observer.contours(i).prostate, observer.image_resolution(i,:));
                metrics.eus.reference_vol(i) = observer.volume(observer.contours(i).eus, observer.image_resolution(i,:));
                metrics.sv.reference_vol(i) = observer.volume(observer.contours(i).sv, observer.image_resolution(i,:));
                metrics.rectum.reference_vol(i) = observer.volume(observer.contours(i).rectum, observer.image_resolution(i,:));
                metrics.bladder.reference_vol(i) = observer.volume(observer.contours(i).bladder, observer.image_resolution(i,:));
                
                metrics.prostate.test_vol(i) = observer.volume(observer2.contours(i).prostate, observer2.image_resolution(i,:));
                metrics.eus.test_vol(i) = observer.volume(observer2.contours(i).eus, observer2.image_resolution(i,:));
                metrics.sv.test_vol(i) = observer.volume(observer2.contours(i).sv, observer2.image_resolution(i,:));
                metrics.rectum.test_vol(i) = observer.volume(observer2.contours(i).rectum, observer2.image_resolution(i,:));
                metrics.bladder.test_vol(i) = observer.volume(observer2.contours(i).bladder, observer2.image_resolution(i,:));
            end
        end
    end
       
    methods(Static)
        function mcc = matthews_corr_coef(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            tps = ref & pred;
            tp = sum(tps(:));
            fps = pred & ~ref;
            fp = sum(fps(:));
            tns = ~ref & ~pred;
            tn = sum(tns(:));
            fns = ref & ~pred;
            fn = sum(fns(:));
            mcc = (tp * tn - fp * fn) / (sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)));
        end
        
        function tps = tps_(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            tps = ref & pred;
            tps = sum(tps(:));
        end
        
        function fps = fps_(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            fps = pred & ~ref;
            fps = sum(fps(:));
        end
        
        function tns = tns_(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            tns = ~ref & ~pred;
            tns = sum(tns(:));
        end
        
        function fns = fns_(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            fns = ref & ~pred;
            fns = sum(fns(:));
        end
        
        function dsc = dice(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            both = ref & pred;
            dsc = 2 * sum(both(:)) / (sum(ref(:)) + sum(pred(:)));
        end
        
        function p = precision(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            both = ref & pred;
            fp = pred & ~ref;
            p = sum(both(:)) / (sum(both(:)) + sum(fp(:)));
        end
        
        function r = recall(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            both = ref & pred;
            fn = ref & ~pred;
            r = sum(both(:)) / (sum(both(:)) + sum(fn(:)));
        end
        
        function vol_diff = absolute_volume_difference(ref, pred)
            v_phys = sum(ref(:));
            v_pred = sum(pred(:));
            vol_diff = (v_phys - v_pred) / v_phys;
        end
        
        function vol = volume(mask, spatial_res)
            vol = sum(mask(:))*prod(spatial_res);
        end
        
        function ji = jaccard(ref, pred)
            ref = logical(ref);
            pred = logical(pred);
            both = ref & pred;
            dsc = 2 * sum(both(:)) / (sum(ref(:)) + sum(pred(:)));
            ji = dsc / (2 - dsc);
        end
    end
end

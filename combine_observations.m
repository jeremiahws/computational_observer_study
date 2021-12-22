function combined = combine_observations(observers)

for i=1:length(observers)
    for j=1:length(observers(i).contours)
        if i == 1
            combined(j).prostate = zeros(size(observers(i).contours(j).prostate), 'uint8');
            combined(j).eus = zeros(size(observers(i).contours(j).eus), 'uint8');
            combined(j).sv = zeros(size(observers(i).contours(j).sv), 'uint8');
            combined(j).rectum = zeros(size(observers(i).contours(j).rectum), 'uint8');
            combined(j).bladder = zeros(size(observers(i).contours(j).bladder), 'uint8');
        end
            
        combined(j).prostate = combined(j).prostate + observers(i).contours(j).prostate;
        combined(j).eus = combined(j).eus + observers(i).contours(j).eus;
        combined(j).sv = combined(j).sv + observers(i).contours(j).sv;        
        combined(j).rectum = combined(j).rectum + observers(i).contours(j).rectum;
        combined(j).bladder = combined(j).bladder + observers(i).contours(j).bladder;
    end        
end
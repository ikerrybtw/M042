% script to read all SVS files and save them
% loops over data directory and calls readSVS
prefix = 'data/';
extension = '.png';
layer = 4;
d = dir(prefix);
destination = 'trainData/';
for i=1:length(d)
    file = d(i).name;
    if file(1) == '.'
        continue;
    else
        readSVS(prefix, file, extension, destination, layer);
    end
    i
end
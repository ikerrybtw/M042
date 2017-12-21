function [ ] = readSVS( prefix, filename, extension, destination, layer )
% read desired layer of an SVS file and save it
I = imread(strcat(prefix,filename), 'index', layer);
lenFile = length(filename);
saveName = strcat(prefix, destination, filename(1:lenFile-4), '_layer', int2str(layer), extension);
imwrite(I, saveName);

end


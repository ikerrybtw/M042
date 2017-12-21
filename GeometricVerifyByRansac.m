function  [tform, inlier_scores, inlier_matches] = ...
    GeometricVerifyByRansac(num,d1,d2,fo,ft,matches,scores,I1,I2)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
matchedPtsOriginal = [fo(1,matches(1,:));fo(2,matches(1,:))];
matchedPtsOriginal = matchedPtsOriginal';
matchedPtsDistorted = [ft(1,matches(2,:)); ft(2,matches(2,:))];
matchedPtsDistorted = matchedPtsDistorted';
[tform,inlierPtsDistorted,inlierPtsOriginal] = estimateGeometricTransform(matchedPtsDistorted,matchedPtsOriginal,'similarity');
flag = 0;
inlier_matches = zeros(2,length(inlierPtsOriginal));
inlier_scores = zeros (1,length(inlierPtsOriginal));
for i=1:length (inlierPtsOriginal)
    for j=1:length(fo)
        if fo(1,j) == inlierPtsOriginal(i,1) && fo(2,j) == inlierPtsOriginal(i,2)
            inlier_matches(1,i)=j;
            for k=1:length(matches)
                if matches(1,k) == j;
                    inlier_matches(2,i) = matches(2,k);
                    inlier_scores(i)=scores(k);
                    flag = 1;
                    break;
                end
            end
        end
        if flag == 1
                flag = 0;
                break;
            end
    end
end
% if num==1
% figure;
% I_estimated = imwarp (I2, tform);
% imshow(I_estimated);
% end

end


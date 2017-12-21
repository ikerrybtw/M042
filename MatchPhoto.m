% this is still in script form, I will write it down as a function later
% global matching

% read CellProfiler output, only necessary in previous version
% cellX = importdata(['TextFiles\HandE\HandE_',num2str(no),'.txt']);
% X = cellX.data;
% cellY = importdata(['TextFiles\TP53\TP53_',num2str(no),'.txt']);
% Y = cellY.data;
% disp('txt files read')


% find center locations from CellProfiler output, again previous version

% [dim1, dim2] = size(rgb2gray(I_x));

% Xpos = X(:,4:5);
% Xpos(:,1) = Xpos(:,1) ;% * dim1;
% Xpos(:,2) = Xpos(:,2) ;% * dim2;

% [dim1, dim2] = size(rgb2gray(I_y));

% Ypos = Y(:,4:5);
% Ypos(:,1) = Ypos(:,1);% * dim1;
% Ypos(:,2) = Ypos(:,2);% * dim2;

% Necessary for SIFT, 3 = scale, 4 = orientation (previous version)

% Xpos(:,3) = round(X(:,15));
% Xpos(:,4) = X(:,18) / 180 * pi;

% Ypos(:,3) = round(Y(:,15));
% Ypos(:,4) = Y(:,18) / 180 * pi;

% change data into single format and find sift features&descriptors

disp('SIFT beginning')
% previous version
% [f_x, d_x] = vl_sift(single(rgb2gray(I_x)), 'frames', Xpos');
% [f_y, d_y] = vl_sift(single(rgb2gray(I_y)), 'frames', Ypos');
% current version, don't use CellProfiler, work directly on image
[f_x, d_x] = vl_sift(single(rgb2gray(I_x)));
[f_y, d_y] = vl_sift(single(rgb2gray(I_y)));
disp('SIFT end')
% find matches between features from H&E and TP53 images

[matches, scores] = vl_ubcmatch(d_x, d_y);
disp('Matching end')
% visualize first 100 matches
VisualizeMatches(scores(1:100),matches(:,1:100),f_x,f_y,d_x,d_y,I_x,I_y);
% find global geometry from matches, assuming they are good matches

initial_matches = matches;
initial_scores = scores;
% eliminate outliers in initial matches using RANSAC, also find transform
[tf, inlier_scores, inlier_matches] = GeometricVerifyByRansac(1, d_x, d_y, f_x, f_y, matches, scores, I_x, I_y);
matches = inlier_matches;
scores = inlier_scores;
% visualize final matches
VisualizeMatches(inlier_scores, inlier_matches, f_x, f_y, d_x, d_y, I_x, I_y);
% can revert back if you want to get rid of RANSAC
% matches = initial_matches;
% scores = initial_scores;

% I also find transform manually using least-squares (they give almost
% exactly the same result)
f_xt = f_x';
f_yt = f_y';
x_pos = f_xt(matches(1,:),1);
x_pos = [x_pos; f_xt(matches(1,:),2)];

y_pos = f_yt(matches(2,:),1:2);
lenY = length(scores);
Y = [y_pos, ones(lenY,1), zeros(lenY, 3)];
Y = [Y; zeros(lenY, 3), y_pos, ones(lenY,1)];

m = pinv(Y) * x_pos;

T = [m(1) m(2) m(3) ; m(4) m(5) m(6) ; 0 0 1];
tform = affine2d(T');
% roughly estimate quality of the found transformation by
% computing l1-distance between real and guess
guess = T(1:2,1:2) * y_pos' + repmat(T(1:2,3), [1, lenY]);
real = f_xt(matches(1,:), 1:2);

diff = real - guess';
sum(diff(:).^2)


% warp TP53 image using the transforms that we found
B = imwarp(I_y, tform);
C = imwarp(I_y, tf);
% calculate rotation angle, scale, translation values from T and check whether
% they make sense or not
s_x = sqrt(m(1)^2 + m(4)^2);
s_y = sqrt(m(2)^2 + m(5)^2);

cos_theta = 0.5 * (m(1) / s_x + m(5) / s_y);
sin_theta = 0.5 * (-m(2) / s_y + m(4) / s_x);

theta = acos(cos_theta) * sign(sin_theta);
t = [m(3);m(6)];

% visualize a random SIFT feature on TP53 image and its projection onto the
% H&E image as given by T
no=20000;
loc=[Y(no,4:5),1];
final_loc=(T*(loc'))';
final_loc=final_loc(1:2);


figure;
imshow(I_y);
hold on
plot(loc(1),loc(2),'b+','MarkerSize',50);


figure;
imshow(I_x);
hold on
plot(final_loc(1),final_loc(2),'b+','MarkerSize',50);


% Y = cellY.data;
% val=sum(abs(Y(:,4:5)-final_loc)');
% gg=find(min(val)==val);
% final_loc
% Y(gg(1),4:5)





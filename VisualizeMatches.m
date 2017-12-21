function result = VisualizeMatches(scores, matches, fa, fb, da, db, Ia, Ib, ExtraNote)

if nargin>8
    ExtraNote = [ExtraNote ' - '];
else
    ExtraNote = [];
end

ScaledScores = scores - min(scores);
ScaledScores = uint8( double(ScaledScores) / max(ScaledScores) * 255).'; 

figure ; clf ;
if size(Ia,1)>size(Ib,1)
   IbScaled = [Ib ; zeros(size(Ia,1)-size(Ib,1),size(Ib,2),size(Ib,3)) ];
   IaScaled=Ia;
else
   IaScaled = [Ia ; zeros(size(Ib,1)-size(Ia,1),size(Ia,2),size(Ia,3)) ]; 
   IbScaled = Ib;
end

imagesc(cat(2, IaScaled, IbScaled)) ;

xa = fa(1,matches(1,:)) ;
xb = fb(1,matches(2,:)) + size(Ia,2) ;
ya = fa(2,matches(1,:)) ;
yb = fb(2,matches(2,:)) ;

hold on ;
h = line([xa ; xb], [ya ; yb]) ;
set(h,'linewidth', 1, 'color', 'w') ;

vl_plotframe(fa(:,matches(1,:))) ;
fb(1,:) = fb(1,:) + size(Ia,2) ;
vl_plotframe(fb(:,matches(2,:))) ;
axis image off ;

title([ExtraNote 'Match Count is ' int2str(size(matches(1,:),2))]);

result = 1;
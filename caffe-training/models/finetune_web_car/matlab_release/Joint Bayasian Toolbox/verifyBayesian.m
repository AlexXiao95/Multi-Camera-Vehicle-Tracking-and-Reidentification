function [scores] = verifyBayesian(model, left_feas, right_feas)

M = inv(model.HE)-inv(model.HI);
[n_pair, d_feat] = size(left_feas);

G = M(1:d_feat,d_feat+1:end);
A = M(d_feat+1:end,d_feat+1:end);

left_feas = bsxfun(@minus,left_feas,model.MU);
right_feas = bsxfun(@minus,right_feas,model.MU);

r1 = (left_feas*A).*left_feas;
r2 = (left_feas*G).*right_feas;
r3 = (right_feas*A).*right_feas;

scores = sum(r1+2*r2+r3,2);

end
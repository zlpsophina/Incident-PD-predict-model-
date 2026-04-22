function [tval, pval] = ttest2_cov_improve(DependentVariable, GroupLabel, Covariate)

Df_E = size(DependentVariable,1) - 2 - size(Covariate,2);
SSE_H = regress_wei(DependentVariable,[ones(size(DependentVariable,1),1),Covariate]);
SSE_H(isnan(SSE_H)) = 1;
% Calulate SSE
[SSE, b] = regress_wei_improve(DependentVariable,[ones(size(DependentVariable,1),1),GroupLabel,Covariate]);
SSE(isnan(SSE)) = 1;

% Calculate F
F = ((SSE_H-SSE)/1)./(SSE./Df_E);
pval =1-fcdf(F,1,Df_E);
tval = sqrt(F).*sign(b(2,:));
end

function [SSE, b] = regress_wei_improve(y,X)
% [b,r,SSE,SSR, T] = y_regress_ss(y,X)
% Perform regression.
% Revised from MATLAB's regress in order to speed up the calculation.
% Input:
%   y - Independent variable.
%   X - Dependent variable.
% Output:
%   b - beta of regression model.
%   r - residual.
%   SSE - The sum of squares of error.
%   SSR - The sum of squares of regression.
%   T - T value for each beta.


b = X \ y;
yhat = X*b;                     % Predicted responses at each data point.
r = y-yhat;                     % Residuals.
SSE=sum(r.^2);
end

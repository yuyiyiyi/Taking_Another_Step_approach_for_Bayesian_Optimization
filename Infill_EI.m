function EI = Infill_EI(x,Kriging_model,fmin)
% get the Kriging prediction and variance
[u,s] = Kriging_Predictor(x,Kriging_model);
% calcuate the EI value
EI = (fmin-u).*Gaussian_CDF((fmin-u)./s)+s.*Gaussian_PDF((fmin-u)./s);

end






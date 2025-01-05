data = readtable("cleaned_data.csv");

X1 = data(:,["HumanDevelopmentIndex_UNDP_", ...
            "AverageTotalYearsOfSchoolingForAdultPopulation_Lee_Lee_2016__Ba", ...
            "PrimaryEnergyConsumptionPerCapita_kWh_person_","GDPPerCapita", ...
            "UrbanPopulation__OfTotalPopulation_", "GiniCoefficient", ...
            "OilProductionPerCapita_kWh_", ...
            "RenewablesPerCapita_kWh_Equivalent_"]);
mdl2 = fitlm(X1,"linear");

X2 = data(:,["HumanDevelopmentIndex_UNDP_", "GDPPerCapita",...
            "AverageTotalYearsOfSchoolingForAdultPopulation_Lee_Lee_2016__Ba", ...
            "PrimaryEnergyConsumptionPerCapita_kWh_person_", ...
            "UrbanPopulation__OfTotalPopulation_", "GiniCoefficient", ...
            "NetImports_TWh_","OilProductionPerCapita_kWh_", ...
            "RenewablesPerCapita_kWh_Equivalent_"]);
mdl1 = fitlm(X2,"linear");

LR = 2*(mdl1.LogLikelihood - mdl2.LogLikelihood);
% has a X2 distribution with a df equals to number of constrained parameters
pval = 1 - chi2cdf(LR, 1);

X1=X1(:, 1:end-1);
X1=X1.Variables;
v= vif(X1);

X3 = data(:,["GDPPerCapita",...
            "AverageTotalYearsOfSchoolingForAdultPopulation_Lee_Lee_2016__Ba", ...
            "PrimaryEnergyConsumptionPerCapita_kWh_person_", ...
            "UrbanPopulation__OfTotalPopulation_", "GiniCoefficient", ...
            "OilProductionPerCapita_kWh_", ...
            "RenewablesPerCapita_kWh_Equivalent_"]);
mdl3 = fitlm(X3,"linear");

X3=X3(:, 1:end-1);
X3=X3.Variables;
v= vif(X3);

X4 = data(:,["GDPPerCapita",...
            "AverageTotalYearsOfSchoolingForAdultPopulation_Lee_Lee_2016__Ba", ...
            "PrimaryEnergyConsumptionPerCapita_kWh_person_",...
            "UrbanPopulation__OfTotalPopulation_", ...
            "OilProductionPerCapita_kWh_", ...
            "RenewablesPerCapita_kWh_Equivalent_"]);

%%X1.GDPPerCapita=log(X1.GDPPerCapita);
%%X1.UrbanPopulation__OfTotalPopulation_=log(X1.UrbanPopulation__OfTotalPopulation_);
%%X1.RenewablesPerCapita_kWh_Equivalent_=log(X1.RenewablesPerCapita_kWh_Equivalent_+0.0001);
%%X4.OilProductionPerCapita_kWh_=log(X4.OilProductionPerCapita_kWh_+0.001);
mdl4 = fitlm(X4,"linear");

X4=X4(:, 1:end-1);
X4=X4.Variables;
v= vif(X4);

LR = 2*(mdl3.LogLikelihood - mdl4.LogLikelihood);
% has a X2 distribution with a df equals to number of constrained parameters
pval = 1 - chi2cdf(LR, 1);
%plotResiduals(mdl4,"fitted")

[T, p, df]= BPtest(mdl4,false);

function [V]=vif(X)
%vif() computes variance inflation coefficients  
%VIFs are also the diagonal elements of the inverse of the correlation matrix [1], 
% a convenient result that eliminates the need to set up the various regressions
%[1] Belsley, D. A., E. Kuh, and R. E. Welsch. Regression Diagnostics. Hoboken, NJ: John Wiley & Sons, 1980.
R0 = corrcoef(X); % correlation matrix
V=diag(inv(R0))';
end

function[T,P,df] = BPtest(z,studentize)
% INPUTS:
% z:            an object of class 'LinearModel' or a (n x p) matrix with the last
%               column corresponding to the dependent (response) variable and the first p-1 columns
%               corresponding to the regressors. Do not include a column of ones for the
%               intercept, this is automatically accounted for.
% studentize:   optional flag (logical). if True the studentized Koenker's statistic is
%               used. If false the statistics from the original Breusch-Pagan test is
%               used.
% OUTPUTS: 
% BP:    test statistics.
% P:     P-value. 
% df:    degrees of freedom of the asymptotic Chi-squared distribution of BP
if nargin == 1
    studentize = true;
end
if isa(z, 'LinearModel')
    
    n  = height(z.Variables);
    df = z.NumPredictors;    
    x = z.Variables{:,z.PredictorNames};
    r = z.Residuals.Raw;
    
else   
    
    x = z(:,1:end-1);
    y = z(:,end);    
    n = numel(y);
    df = size(x,2);    
    lm = fitlm(x,y);
    r = lm.Residuals.Raw;    
    
end
aux = fitlm(x,r.^2);
T = aux.Rsquared.Ordinary*n;
if ~studentize
    lam = (n-1)/n*var(r.^2)/(2*((n-1)/n*var(r)).^2);
    T  = T*lam;
end
P = 1-chi2cdf(abs(T),df);
end



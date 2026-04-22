clear all
close all



addpath(genpath('MediationToolbox-master\MediationToolbox-master'))

load ukb_prot_tables.mat

load ../ukb_bl_dataset.mat

ukb_bl_FieldID=ukb_bl_data(:,[2:end]).Properties.VariableNames';


ukb_bl_FieldID2=nan(length(ukb_bl_FieldID),1);
ukb_bl_name2=[];
for i=1:length(ukb_bl_FieldID)
    t_names= ukb_bl_FieldID{i};
    t1_names=split(t_names,'_');
    t2_names= t1_names{1};
    t3_names=str2num(t2_names(2:end));
    ukb_bl_FieldID2(i,1)=t3_names;

    ind2=ukb_bl_name.FieldID==t3_names;
    num(i,1)= sum(ind2);
    ukb_bl_name2=[ukb_bl_name2;ukb_bl_name(ind2,:)];
end

unique(ukb_bl_name.FieldID);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

incident_pd_ttest_table=readtable('../incident_pd_ttest_table.csv');
corrected_pval=incident_pd_ttest_table.BF_p;


ukb_bl_name2=ukb_bl_name2(corrected_pval<0.05,:);
corrected_pval_2=corrected_pval(corrected_pval<0.05,:);

ind2=corrected_pval<0.05;
ind2=[1;ind2];
ukb_bl_data=ukb_bl_data(:,ind2>0);




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




ukb_cov_info=readtable('../ukb_cov_info.csv');
ukb_cov_info.Smoking=ukb_cov_info.Smoking+1;
ukb_cov_info.Alcohol=ukb_cov_info.Alcohol+1;


matchit_full_data=readtable('UKB_prot_incident_PD_matchit_nearest_data.csv');

%%


ukb_eid = intersect(intersect(intersect(matchit_full_data.eid,ukb_cov_info.eid),ukb_prot_tables.eid),ukb_bl_data.eid);


[~, ind2] = intersect(ukb_prot_tables.eid,ukb_eid);
ukb_prot_tables=ukb_prot_tables(ind2,:);


[~, ind2] = intersect(ukb_cov_info.eid,ukb_eid);
ukb_cov_info=ukb_cov_info(ind2,:);

[~, ind2] = intersect(matchit_full_data.eid,ukb_eid);
matchit_full_data=matchit_full_data(ind2,:);


[~, ind2] = intersect(ukb_bl_data.eid,ukb_eid);
ukb_bl_data=ukb_bl_data(ind2,:);

[~, ind2] = intersect(ukb_cov_info.eid,ukb_bl_data.eid);
ukb_cov_info=ukb_cov_info(ind2,:);

ukb_2_eid=[matchit_full_data.eid,ukb_cov_info.eid,ukb_bl_data.eid,ukb_prot_tables.eid];



%% %%%%%%ukb_cov_info.sex=dummyvar(ukb_cov_info.sex);%ukb_cov_info.centre=cell2num(ukb_cov_info.centre);

ukb_cov_info.age=zscore(ukb_cov_info.age);
ukb_cov_info.education=zscore(ukb_cov_info.education);
ukb_cov_info.Townsend=zscore(ukb_cov_info.Townsend);
ukb_cov_info.BMI=zscore(ukb_cov_info.BMI);


%ukb_cov_info.group=strcmp(matchit_full_data.group,'TRUE');



% ind2=ukb_cov_info.duration<25;
% ukb_cov_info=ukb_cov_info(ind2,:);
unique(ukb_cov_info.Ethnic)
unique(ukb_cov_info.centre)

centre_num = unique(ukb_cov_info.centre);
for i=1:length(centre_num)
    num_each_size(i) =  sum(ukb_cov_info.centre==centre_num(i));
end
bl_centre = zeros(length(ukb_cov_info.centre),sum(num_each_size>20));
for i=1:length(centre_num)
    if num_each_size(i)>10
        bl_centre(ukb_cov_info.centre==centre_num(i),i) = 1;
    end
end
index = sum(bl_centre);
bl_centre(:,index==0) = [];

centre_names=[];
for i=1:size(bl_centre,2)
    centre_names{i}=['centre',num2str(i)];
end

bl_centre_table=array2table(bl_centre,'VariableNames',centre_names);

ukb_cov_info=[ukb_cov_info,bl_centre_table(:,[1:end-1])];

%%Ethnic%%%%%%ukb_cov_info.sex=dummyvar(ukb_cov_info.sex);%ukb_cov_info.Ethnic=cell2num(ukb_cov_info.Ethnic);
%ukb_cov_info.Ethnic=cell2num(ukb_cov_info.Ethnic);

Ethnic_num = unique(ukb_cov_info.Ethnic);
for i=1:length(Ethnic_num)
    num_each_size(i) =  sum(ukb_cov_info.Ethnic==Ethnic_num(i));
end
bl_Ethnic = zeros(length(ukb_cov_info.Ethnic),sum(num_each_size>20));
for i=1:length(Ethnic_num)
    if num_each_size(i)>10
        bl_Ethnic(ukb_cov_info.Ethnic==Ethnic_num(i),i) = 1;
    end
end
index = sum(bl_Ethnic);
bl_Ethnic(:,index==0) = [];

Ethnic_names=[];
for i=1:size(bl_Ethnic,2)
    Ethnic_names{i}=['Ethnic',num2str(i)];
end

sum(sum(bl_Ethnic))

bl_Ethnic_table=array2table(bl_Ethnic,'VariableNames',Ethnic_names);
ukb_cov_info=[ukb_cov_info,bl_Ethnic_table(:,[1:end-1])];


% %

Smoking_num = unique(ukb_cov_info.Smoking);
for i=1:length(Smoking_num)
    num_each_size(i) =  sum(ukb_cov_info.Smoking==Smoking_num(i));
end
bl_Smoking = zeros(length(ukb_cov_info.Smoking),sum(num_each_size>20));
for i=1:length(Smoking_num)
    if num_each_size(i)>10
        bl_Smoking(ukb_cov_info.Smoking==Smoking_num(i),i) = 1;
    end
end
index = sum(bl_Smoking);
bl_Smoking(:,index==0) = [];

Smoking_names=[];
for i=1:size(bl_Smoking,2)
    Smoking_names{i}=['Smoking',num2str(i)];
end

sum(sum(bl_Smoking))
bl_Smoking_table=array2table(bl_Smoking,'VariableNames',Smoking_names);

ukb_cov_info=[ukb_cov_info,bl_Smoking_table(:,[1:2])];


% %
Alcohol_num = unique(ukb_cov_info.Alcohol);
for i=1:length(Alcohol_num)
    num_each_size(i) =  sum(ukb_cov_info.Alcohol==Alcohol_num(i));
end
bl_Alcohol = zeros(length(ukb_cov_info.Alcohol),sum(num_each_size>20));
for i=1:length(Alcohol_num)
    if num_each_size(i)>10
        bl_Alcohol(ukb_cov_info.Alcohol==Alcohol_num(i),i) = 1;
    end
end
index = sum(bl_Alcohol);
bl_Alcohol(:,index==0) = [];

Alcohol_names=[];
for i=1:size(bl_Alcohol,2)
    Alcohol_names{i}=['Alcohol',num2str(i)];
end

sum(sum(bl_Alcohol))
bl_Alcohol_table=array2table(bl_Alcohol,'VariableNames',Alcohol_names);

ukb_cov_info=[ukb_cov_info,bl_Alcohol_table(:,[1:2])];


%%


Covariate_names=ukb_cov_info(:,[2,3,6:8,11:end]).Properties.VariableNames;
Covariate_ukb=ukb_cov_info(:,[2,3,6:8,11:end]).Variables;



GroupLabel=strcmp(matchit_full_data.group,'TRUE');
ttest_results=nan(size(ukb_bl_name2,1) ,6);

r_blood=[];
pval=[];

X=double(GroupLabel);
Y=ukb_prot_tables.NEFL;

pd_Y=Y(X==1,:);
pd_Z=Covariate_ukb(X==1,:);


blood_stats=[];
blood_r_ci_p=[];
blood_mediation_results=[];

for i =1:size(ukb_bl_name2,1)  %147
    tic

    id=i+1;
    t_data=ukb_bl_data(:,id).Variables;
    ukb_bl_name2(i,3)=ukb_bl_data(:,id).Properties.VariableNames;
    M=t_data;

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pd_M=M(X==1,:);

    tt=[pd_Y,pd_M];
    ind2=~isnan(sum(tt'));

    x1=pd_Y(ind2,:);
    y1=pd_M(ind2,:);
    z1=pd_Z(ind2,:);

    [r_blood(i,1),pval(i,1)]=partialcorr(x1,y1,z1);

    mycorr = @(x1,y1,z1) partialcorr(x1,y1,z1);
    nIterations = 100;
    [tt_ci, ~] = bootci(nIterations,{mycorr,x1,y1,z1});
    tt_ci=round(tt_ci,3);


    r=round(r_blood(i,1),3);
    blood_r_ci_p{i,1}=strcat([num2str(r),' (' , num2str(tt_ci(1)),' to ' num2str(tt_ci(2)),')'   ]);
    blood_r_ci_p{i,2}=pval(i,1);
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    ind2=sum(~isnan([X,Y,M]'))>2;
    ind2=ind2';
    t_mediation=[];
    [paths, stats] = mediation(X(ind2), Y(ind2), M(ind2), 'covs',Covariate_ukb(ind2,:), 'doCIs' , 'bootsamples', 1000);%'plots', 'verbose', 'robust',
    %[paths, stats] = mediation(X(ind2), Y(ind2), M(ind2), 'plots', 'verbose');
    % mediation_path_diagram(stats)

    t_ci=stats.ci;
    t_mediation=[stats.paths;stats.ste;stats.p;t_ci(:,:,1);t_ci(:,:,2)];
    t_mediation_table=  array2table( t_mediation,"RowNames",{'paths','ste','pvalue','CI1','CI2'},"VariableNames",stats.names);

    blood_stats{i}=stats;
    blood_mediation_results{i}=t_mediation_table;
    % blood_CIs{i}=stats.ci;

    toc
end


%r_table=  array2table([r_blood,pval], "RowNames", ukb_bl_name2.Description,"VariableNames" ,{'r','p'});,
r_table=  array2table(blood_r_ci_p, "RowNames", ukb_bl_name2.Description,"VariableNames" ,{'r','p'});,
writetable(r_table,'pd_nefl_corr_blood.csv','WriteRowNames',true);%'WriteRowNames',falseukb_bl_name2

save blood_mediation_result   blood_mediation_results blood_stats










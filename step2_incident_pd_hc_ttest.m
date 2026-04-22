clear all
close all


load ukb_bl_dataset.mat

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


%%

ukb_cov_info=readtable('../preclincal_pd/ukb_cov_info.csv');
ukb_cov_info.Smoking=ukb_cov_info.Smoking+1;
ukb_cov_info.Alcohol=ukb_cov_info.Alcohol+1;



matchit_full_data=readtable('../preclincal_pd/UKB_incident_PD_matchit_nearest_data.csv');


[~, ind2] = intersect(ukb_cov_info.eid,matchit_full_data.eid);
ukb_cov_info=ukb_cov_info(ind2,:);

[~, ind2] = intersect(matchit_full_data.eid,ukb_cov_info.eid);
matchit_full_data=matchit_full_data(ind2,:);




[~, ind2] = intersect(ukb_bl_data.eid,ukb_cov_info.eid);
ukb_bl_data=ukb_bl_data(ind2,:);

[~, ind2] = intersect(ukb_cov_info.eid,ukb_bl_data.eid);
ukb_cov_info=ukb_cov_info(ind2,:);

ukb_cov_info.age=zscore(ukb_cov_info.age);
ukb_cov_info.education=zscore(ukb_cov_info.education);
ukb_cov_info.Townsend=zscore(ukb_cov_info.Townsend);
ukb_cov_info.BMI=zscore(ukb_cov_info.BMI);


%ukb_cov_info.group=strcmp(matchit_full_data.group,'TRUE');



% ind2=ukb_cov_info.duration<25;
% ukb_cov_info=ukb_cov_info(ind2,:);

unique(ukb_cov_info.Ethnic)
unique(ukb_cov_info.centre)



%% 

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
ukb_eid=[matchit_full_data.eid,ukb_cov_info.eid,ukb_bl_data.eid];

Covariate_names=ukb_cov_info(:,[2,3,6:8,11:end]).Properties.VariableNames;
Covariate_ukb=ukb_cov_info(:,[2,3,6:8,11:end]).Variables;

GroupLabel=strcmp(matchit_full_data.group,'TRUE');


ttest_results=nan(size(ukb_bl_name2,1) ,6);

pp2=[];
for i =1:size(ukb_bl_name2,1)  %147
    tic

    id=i+1;
    t_data=ukb_bl_data(:,id).Variables;
    ukb_bl_name2(i,3)=ukb_bl_data(:,id).Properties.VariableNames;


    ttest_results(i,1)=nanmean(t_data(GroupLabel==0,:));
    ttest_results(i,2)=nanstd(t_data(GroupLabel==0,:));

    ttest_results(i,3)=nanmean(t_data(GroupLabel==1,:));
    ttest_results(i,4)=nanstd(t_data(GroupLabel==1,:));

    ind2=~isnan(t_data);

    [tval, pval] = ttest2_cov_improve(t_data(ind2,:), GroupLabel(ind2,:), Covariate_ukb(ind2,:));

    ttest_results(i,5)=tval;
    ttest_results(i,6)=pval;
    disp([num2str(pval,'%.16g')])

    pp2{i,1} = [num2str(pval,'%.16g')];

    toc
end



ttest_table=  array2table( ttest_results,'VariableNames',{'HC_mean','HC_std','PD_mean','PD_STD',....
    'tstats','pvalue'},'RowNames',ukb_bl_name2.Description);%


corrected_p=mafdr(ttest_results(:,6),'BHFDR', true);
ttest_table.corrected_p=corrected_p;

ttest_table.BF_p=ttest_results(:,6) .* length(ttest_results);

%writetable(ttest_table,'incident_pd_ttest_table.csv','WriteRowNames',true);%'WriteRowNames',falseukb_bl_name2



%%
incident_pd_ttest_table=readtable('../preclincal_pd/incident_pd_ttest_table.csv');

incident_pd_ttest_cell=[];
for i =1:size(incident_pd_ttest_table,1)  %147

    tt_name=incident_pd_ttest_table(i,1).Row;
    incident_pd_ttest_cell{i,1}=tt_name{1};


    a_m1=round(incident_pd_ttest_table(i,2).Variables,1);
    a_std=round(incident_pd_ttest_table(i,3).Variables,1);
    incident_pd_ttest_cell{i,2}=strcat([num2str(a_m1),' (' num2str(a_std),')'   ]);


    b_m1=round(incident_pd_ttest_table(i,4).Variables,1);
    b_std=round(incident_pd_ttest_table(i,5).Variables,1);
    incident_pd_ttest_cell{i,3}=strcat([num2str(b_m1),' (' num2str(b_std),')'   ]);

    incident_pd_ttest_cell{i,4}=round(incident_pd_ttest_table(i,6).Variables,1);

    incident_pd_ttest_cell{i,5}=round(incident_pd_ttest_table(i,7).Variables,3);
    incident_pd_ttest_cell{i,6}=round(incident_pd_ttest_table(i,9).Variables,3);

end

incident_pd_ttest_cell=table(incident_pd_ttest_cell);
writetable(incident_pd_ttest_cell,'results/incident_pd_ttest_cell.csv','WriteRowNames',true);%'WriteRowNames',falseukb_bl_name2



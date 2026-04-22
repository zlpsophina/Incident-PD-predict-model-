clear  all
close all



load ukb_bl_dataset.mat

ukb_bl_FieldID=ukb_bl_data(:,[2:end]).Properties.VariableNames;


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



incident_pd_ttest_table=readtable('../preclincal_pd/incident_pd_ttest_table.csv');
corrected_pval=incident_pd_ttest_table.BF_p;

ukb_bl_name2.incident_pd_names=incident_pd_ttest_table.Row;
ukb_bl_name2.ukb_bl_data_id=ukb_bl_data(:,[2:end]).Properties.VariableNames';

ukb_bl_name2=ukb_bl_name2(corrected_pval<0.05,:);



corrected_pval_2=corrected_pval(corrected_pval<0.05,:);

ind2=corrected_pval<0.05;
ind2=[1;ind2];
ukb_bl_data=ukb_bl_data(:,ind2>0);


for  i=1:size(ukb_bl_name2,1)

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Creatinine (enzymatic) in urine','Creatinine in urine');

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Diastolic blood pressure, automated reading','Diastolic blood pressure');
    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Systolic blood pressure, automated reading','Systolic blood pressure');

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Pulse rate, automated reading','Pulse rate');
    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Glycated haemoglobin (HbA1c)','HbA1c');

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Red blood cell (erythrocyte) count','RBC');

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Alkaline phosphatase','ALP');
    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Gamma glutamyltransferase','GGT');

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Red blood cell (erythrocyte) distribution width','RDW');

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'LDL direct','LDL');

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Platelet crit' ,'PCT' );

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Mean reticulocyte volume' ,'MRV' );

    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Haemoglobin concentration' ,'HGB' );
    ukb_bl_name2{i,2}=strrep(ukb_bl_name2{i,2},'Haematocrit percentage' ,'HCT' );

end


ukb_bl_data.Properties.VariableNames=[{'eid'};ukb_bl_name2.Description];

%%
load ../preclincal_pd\pd_hc_cov_data.mat

ukb_bl_data_cohend=[];
hc_bl_cov=[];pd_bl_cov=[];
for i =1:length(different_hc_cov)

    hc_cov=different_hc_cov{i};
    pd_cov=different_pd_cov{i};

    hc_bl_cov=[hc_bl_cov;hc_cov];
    pd_bl_cov=[pd_bl_cov;pd_cov];

end

duration_data=[hc_bl_cov;pd_bl_cov];



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



[~, ind2] = intersect(duration_data.eid,ukb_cov_info.eid);
duration_data=duration_data(ind2,:);


ukb_eid=[matchit_full_data.eid,ukb_cov_info.eid,ukb_bl_data.eid,duration_data.eid];

%% %%%%%%ukb_cov_info.sex=dummyvar(ukb_cov_info.sex);%ukb_cov_info.centre=cell2num(ukb_cov_info.centre);

centre_num = unique(ukb_cov_info.centre);
for i=1:length(centre_num)
    num_each_size(i) =  sum(ukb_cov_info.centre==centre_num(i));
end
bl_centre = zeros(length(ukb_cov_info.centre),sum(num_each_size>100));

for i=1:length(centre_num)
    if num_each_size(i)>100
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

%bl_centre_table.centre2=ukb_cov_info.centre;   check

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


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
ukb_eid=[matchit_full_data.eid,ukb_cov_info.eid,ukb_bl_data.eid,duration_data.eid];

Covariate_names=ukb_cov_info(:,[2,3,6:8,11:end]).Properties.VariableNames;
Covariate_ukb=ukb_cov_info(:,[2,3,6:8,11:end]).Variables;

GroupLabel=strcmp(matchit_full_data.group,'TRUE');



hcpd_regressed_data=[];
for i =1:size(ukb_bl_name2,1)  %147
    tic

    id=i+1;
    t_data=ukb_bl_data(:,[1,id]).Variables;
    ukb_bl_name2(i,5)=ukb_bl_data(:,id).Properties.VariableNames;

    ind2=~isnan(t_data(:,2));
    [T,Res,Res_with_cons,beta]=BWAS_Tregression(Covariate_ukb(ind2,:),t_data(ind2,2));

    [T2,~,Res_with_cons2,beta,beta_ci]=BWAS_Tregression2(Covariate_ukb(ind2,:),t_data(ind2,2));


    t_data=t_data(ind2,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    eid_num=size(t_data,1);
    target_num(id)=eid_num;

    t_marker=nan(size(ukb_bl_data,1),3);
    hcpd_eid=ukb_bl_data.eid;
    for i=1:size(t_data,1)
        ind2=hcpd_eid== t_data(i,1);
        if sum(ind2)>0
            t_marker(ind2,1)=t_data(i,2);
            t_marker(ind2,2)=Res_with_cons(i);
            t_marker(ind2,3)=t_data(i,1);
        end
    end
    hcpd_regressed_data=[hcpd_regressed_data,t_marker(:,2)];

    toc
end


hcpd_regressed_data2=[hcpd_eid,hcpd_regressed_data];%matchit_full_data.group

hcpd_regressed_table=  array2table( hcpd_regressed_data2,'VariableNames',[{'eid'};ukb_bl_name2.Description]);%

hcpd_regressed_table.duration=duration_data.duration;
hcpd_regressed_table.group=matchit_full_data.group;


writetable(hcpd_regressed_table,'hcpd_regressed_table.csv','WriteRowNames',true);%'WriteRowNames',falseukb_bl_name2








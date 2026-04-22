close all
clear all



UKB_incident_PD=readtable('UKB_incident_PD.csv');

matchit_full_data=readtable('UKB_incident_PD_matchit_nearest_data.csv');


[~, ind2] = intersect(matchit_full_data.eid,UKB_incident_PD.eid);
PD_matchit_full_data=matchit_full_data(ind2,:);


uniq_subclass=unique(PD_matchit_full_data.subclass);


HC_eid=[];
for i=1:length(uniq_subclass)

    ind1 = matchit_full_data.subclass==uniq_subclass(i);

    t_ind1=matchit_full_data(ind1,:);

    ind2=strcmp(t_ind1.group,'FALSE');
    t_ind2=t_ind1(ind2,:);
    HC_eid=[HC_eid;t_ind2.eid];
end

hc_pd_eid=[HC_eid;PD_matchit_full_data.eid];

[~, ind2] = intersect(matchit_full_data.eid,hc_pd_eid);
hc_pd_matchit_full_data=matchit_full_data(ind2,:);




%%

load  hcpd_regressed_data.mat


boxplot(hcpd_regressed_table.("Apolipoprotein B"),  hcpd_regressed_table.group  )


[~, ind2] = intersect(hcpd_regressed_table.eid,hc_pd_matchit_full_data.eid);
hcpd_regressed_table=hcpd_regressed_table(ind2,:);


[~, ind2] = intersect(hc_pd_matchit_full_data.eid,hcpd_regressed_table.eid);
hc_pd_matchit_full_data=hc_pd_matchit_full_data(ind2,:);


ukb_eid=[hc_pd_matchit_full_data.eid,hcpd_regressed_table.eid];

GroupLabel=strcmp(hc_pd_matchit_full_data.group,'TRUE');


sum(GroupLabel==1) 


%%



load pd_hc_cov_data.mat


FO_PD_HC_cohend=nan(65,7);
FO_PD_HC_mean=nan(65,7);

for i =1:length(different_hc_cov)

    hc_cov=different_hc_cov{i};
    pd_cov=different_pd_cov{i};


    [~, ind2] = intersect(hcpd_regressed_table.eid,hc_cov.eid);
    hc_data=hcpd_regressed_table(ind2,[2:end-2]).Variables;


    [~, ind2] = intersect(hcpd_regressed_table.eid,pd_cov.eid);
    pd_data=hcpd_regressed_table(ind2,[2:end-2]).Variables;

    t_cohend=[];t_mean=[];
    for k=1:size(pd_data,2)
        t_cohend(k)=computeCohen_d(pd_data(:,k),hc_data(:,k),'independent');
        t_mean(k)=nanmean(pd_data(:,k))-nanmean(hc_data(:,k));

    end

    FO_PD_HC_mean(:,i)=t_mean;
    FO_PD_HC_cohend(:,i)=t_cohend;

end



ukb_years_cohend_table= array2table( FO_PD_HC_cohend,'VariableNames',...
    {'-14 years','-10 years','-8 years','-6 years','-4 years','-2 years','0 years'},...
    'RowNames',hcpd_regressed_table(:,[2:end-2]).Properties.VariableNames);%'HC','PD','corrected_p'

ukb_years_mean_table= array2table( FO_PD_HC_mean,'VariableNames',...
    {'-14 years','-10 years','-8 years','-6 years','-4 years','-2 years','0 years'},...
    'RowNames',hcpd_regressed_table(:,[2:end-2]).Properties.VariableNames);%'HC','PD','corrected_p'




%writetable(ukb_years_cohend_table,'results/ukb_years_cohend_table.csv','WriteRowNames',true);%'WriteRowNames',falseukb_bl_name2


%%

t_names=ukb_years_cohend_table.Properties.RowNames;


incident_pd_ttest_table=readtable('incident_pd_ttest_table.csv');
corrected_pval=incident_pd_ttest_table.BF_p;

ukb_bl_name2.incident_pd_names=incident_pd_ttest_table.Row;
ukb_bl_name2.cohen_names=ukb_years_cohend_table.Properties.RowNames;;


ukb_bl_name3=ukb_bl_name2(corrected_pval<0.05,:);
data=ukb_years_cohend_table(corrected_pval<0.05,:);




[c_data,ind2]=sort(data.("0 years"),'descend');

cohend_names= ukb_bl_name3(ind2,:);
FO_time_interval_cohend=data(ind2,:);

save FO_time_interval_cohend_data FO_time_interval_cohend cohend_names
%%

data=FO_time_interval_cohend;


figure(1);
imagesc(data.Variables);

% 添加X轴和Y轴标签
xlabel('Years to diagnosis', 'FontSize', 12);
%ylabel('Y轴名称', 'FontSize', 12);

% 添加标题
title("All incident PD vs Control", 'FontSize', 14);

% 添加颜色条
colorbar;

% 设置坐标轴刻度和标签（可选）
xticks(1:7);
xticklabels({'<-12','-12 to -10','-10 to -8','-8 to -6','-6 to -4','-4 to -2','-2 to 0'});
yticks(1:25);
yticklabels(cohend_names.cohen_names);

set(gcf,'position',[300 300 550 600])


% 添加颜色条
colorbar;
colormap(parula);

% 设置 colorbar 的范围以确保 0 在中间
data=data.Variables;
caxis([-max(abs(data(:))) max(abs(data(:)))]);

% 额外确认字体（保险）
set(gca,'FontName','Arial');


%saveas(figure(1),['', fig_title], 'fig');
print(figure(1), '-dtiff','-r300', ['results/FO_PD_time_cohend.tiff'])





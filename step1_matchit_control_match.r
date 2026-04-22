rm(list=ls())
library("MatchIt")
library("optmatch")
library("marginaleffects")


ukb_cov_info <- read.csv("ukb_cov_info.csv", header=TRUE)
ukb_cov_info<-na.omit(ukb_cov_info[,1:10])



ukb_cov_info$sex<-as.factor(ukb_cov_info$sex)
ukb_cov_info$Ethnic<-as.factor(ukb_cov_info$Ethnic)
ukb_cov_info$centre<-as.factor(ukb_cov_info$centre)



hc_eid <- read.csv("UKB_control_eid.csv", header=TRUE)
#pd_eid <-read.csv("UKB_incident_PD.csv", header=TRUE)
pd_eid <-read.csv("UKB_dignosised_PD.csv", header=TRUE)


index <- ukb_cov_info$eid %in% hc_eid$eid
hc_covinfo<-ukb_cov_info[index,]

hc_n<-nrow(hc_covinfo)
hc_group<-data.frame(rep(FALSE,times=hc_n))
names(hc_group)<-"group"
hc_covinfo<-cbind(hc_covinfo,hc_group)


index <- ukb_cov_info$eid %in% pd_eid$eid
pd_covinfo<-ukb_cov_info[index,]

pd_n<-nrow(pd_covinfo)
pd_group<-data.frame(rep(TRUE,times=pd_n))
names(pd_group)<-"group"
pd_covinfo<-cbind(pd_covinfo,pd_group)


pd_hc_covinfo<-rbind(pd_covinfo,hc_covinfo)



m.out1 <- matchit(group~ education +Townsend + BMI,
                  method = "nearest", distance = "mahalanobis", ratio = 3,  link = "logit" ,  #"mahalanobis"   "glm"
                  exact = ~ age + sex+ Ethnic+Smoking+Alcohol,
                  data = pd_hc_covinfo)  # 

m.out1

# Checking balance after NN matching
summary(m.out1, un = FALSE)

#plot(m.out1, type = "jitter", interactive = FALSE)

#plot(m.out1, type = "density", interactive = FALSE,
 #    which.xs = ~age + sex + education+Townsend + BMI)


m.data <- match.data(m.out1)


# 画箱线图
boxplot(m.data$age~m.data$group, main="Boxplot", ylab="age", xlab="group")


write.csv(m.data,"UKB_dignosised_PD_matchit_nearest_data.csv")  #"glm"  logit   exact = ~ age + sex + Ethnic,
#write.csv(m.data,"UKB_incident_PD_matchit_nearest_data.csv")  #"glm"  logit   exact = ~ age + sex + Ethnic+ centre,group~ education +Townsend + BMI,








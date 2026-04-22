# 清空环境变量（和Matlab的 rm(list=ls()) 一样）
rm(list = ls())



# 加载ggplot2绘图包
library(ggplot2)

# ===================== 1. 读取数据 =====================
# 读取回归校正后的数据
hcpd_regressed_data <- read.csv("hcpd_regressed_table.csv", header = TRUE)

# 查看数据前1行前10列，确认数据格式
head(hcpd_regressed_data[1, 1:10])

# 把分组变量转换成因子型（分组绘图必须）
hcpd_regressed_data$group <- factor(hcpd_regressed_data$group)

# 简化数据名，方便后续使用
my_data <- hcpd_regressed_data

# 获取列名，并把点号 . 替换成空格（让坐标轴标题更美观）
ukb_bl_name <- colnames(my_data)
ukb_bl_name <- gsub("\\.", " ", ukb_bl_name)


# 只保留 X轴范围 -15 ~ 0 内的数据
my_data_clean <- my_data[my_data$duration >= -15 & my_data$duration <= 0 & !is.na(my_data$duration), ]


# ===================== 2. 设置输出图片路径（你要的文件夹） =====================
# 目标保存路径：AGO-figs 文件夹
fig_output_path <- "D:/华为家庭存储/I-盘/ukb/preclincal_pd/source_AGO/AGO-figs"

# 如果文件夹不存在，自动创建（避免报错）
if (!dir.exists(fig_output_path)) {
  dir.create(fig_output_path, recursive = TRUE)
}

# ===================== 3. 循环绘制65张趋势图 =====================
# 循环绘制第 2~66 列（共65个指标）
for (i in 1:65) {
  
  # 当前指标的名称
  y_name <- ukb_bl_name[i + 1]
  
  # ===================== 绘制LOESS平滑趋势图 =====================
  p_fig <- ggplot(data = my_data,
                  aes(x = duration,          # X轴：距诊断年数
                      y = my_data[, 1 + i],  # Y轴：当前血液指标
                      group = group,         # 按group分组
                      color = group)) +      # 按group着色
    
    # LOESS平滑曲线（临床趋势图常用）
    geom_smooth(method = 'loess', 
                span = 2, 
                se = TRUE,        # 显示置信区间
                size = 1, 
                alpha = 0.22, 
                show.legend = FALSE) +
    
    # 分组颜色设置
    scale_color_manual(values = c("#607090", "#E59A21")) +  
    
    # 置信区间填充色
    scale_fill_manual(values = c("grey", "grey")) +
    
    # X轴设置：-15到0年，间隔1年
    scale_x_continuous(name = "Years to diagnosis", 
                       breaks = seq(-15, 0, by = 1),  
                       limits = c(-15, 0)) +
    
    # Y轴名称：当前指标名
    scale_y_continuous(name = y_name) +
    
    # 主题：清晰无背景
    theme_classic() +
    
    # 字体大小、不显示图例
    theme(axis.text = element_text(size = 10),
          legend.position = "none", 
          plot.title = element_text(hjust = 0.5))
  
  # 在控制台打印图片（可选）
  print(p_fig)
  
  # ===================== 4. 保存图片到 AGO-figs 文件夹 =====================
  # 处理文件名：去掉非法字符 / \ : * ? " < > | 避免保存失败
  safe_y_name <- gsub("[^a-zA-Z0-9_ ]", "", y_name)
  fig_name <- paste0(safe_y_name, "_plot.tiff")
  
  # 保存 tiff 格式图片到指定目录
  ggsave(
    filename = fig_name,
    plot = p_fig,
    path = fig_output_path,  # 强制保存到 AGO-figs
    width = 11, 
    height = 5, 
    units = "cm",
    dpi = 300  # 高清输出
  )
}
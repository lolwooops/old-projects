library(ggplot2)
wcs=data.frame(X493tp_data2)
wcsg = ggplot(wcs,aes(Date,WCS))+geom_point(color="lightcoral")+geom_line(color="grey0")
wcsg+
  ylab("WCS Prices (US$/B)")+
  xlab("Date")+
  ggtitle("Figure 1. WCS Price")
summary(wcs)
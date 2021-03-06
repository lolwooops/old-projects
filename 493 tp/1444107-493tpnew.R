library(fpp2)
library(urca)
X493tpdata = read_excel("C:/Users/Tim/Desktop/493tpdata.xlsx")

data = data.frame(X493tpdata)
datats = ts(data,start=c(2009,1),end=c(2018,8),frequency=12)
summary(datats)
wcsts = lm(WCS ~ WTI + MAYA + EXCH + PADD2 + PADD3 + Usimports, data = data)
checkresiduals(wcsts)
wcsres = resid(wcsts)
ggtsdisplay(wcsres)
checkresiduals(wcsres)

##residuals
WCSBC = BoxCox(wcsrests,lambda=1/3)
ggtsdisplay(WCSBC)
WCSBC = BoxCox(wcsrests,lambda=1/3)+4
WCSBCTS = ts(WCSBC, start = c(2009,1), end = c(2018, 8), frequency = 12)
ggtsdisplay(WCSBCTS, main = "BoxCox'd WCS residuals")
summary(ur.df(WCSBCTS, selectlags=c("Fixed")))

WCSAA = auto.arima(WCSBCTS)
WCSA1 = Arima(WCSBCTS, order=c(1,0,1))
#comparing forecasts of the 2 ARIMA models
WCSAAFC = forecast(WCSAA, h=24)
autoplot(WCSAAFC)
WCSA1FC = forecast(WCSA1, h=24)
autoplot(WCSA1FC)

y = Arima(datats[,2],order=c(1,0,0),xreg=datats[,3:8])
y2 = Arima(datats[,2],order=c(1,0,1),xreg=datats[,3:8])
autoplot(forecast(y,xreg=datats[,3:8],h=10))+ylab("WCS prices") #produces up to 2025 - not what I wanted
autoplot(forecast(y,xreg=datats[,3:8],h=5))+ylab("WCS prices") #test
autoplot(forecast(y,xreg=datats[,3:8],h=1))+ylab("WCS prices") #test
autoplot(forecast(y,xreg=datats[,3:8],h=25))+ylab("WCS prices") #test
autoplot(forecast(y2,xreg=datats[,3:8]), h=10)+ylab("WCS prices") #test
checkresiduals(y)
checkresiduals(y2)
y3=auto.arima(datatsd[,2],xreg=datatsd[,3:8])
summary(y3)

summary(y2)

auto.arima(diff(datats[,2]))
auto.arima(datats[,2])
checkresiduals(diff(datats[,2]))

ynaive = naive(datats[,2],12)
autoplot(ynaive)+ylab("WCS Prices")
ynaive1 = naive(datats[,2],1)
summary(ynaive1)

library(vars)
X493tpdatavar <- read_excel("C:/Users/Tim/Desktop/493tpdatavar.xlsx")
data2 = data.frame(X493tpdatavar)
datats2 = ts(data2,start=c(2009,1),end=c(2018,8),frequency=12)

VARselect(datats2[,2:5], lag.max=8,
          type="const")[["selection"]]

var1 <- VAR(datats2[,2:5], p=1, type="const")
serial.test(var1, lags.pt=10, type="PT.asymptotic")
var2 <- VAR(datats2[,2:5], p=2, type="const")
serial.test(var2, lags.pt=10, type="PT.asymptotic")

varfc = forecast(var2, h=12)
varfc
summary(varfc)
autoplot(varfc)

VARselect(datats2[,2:4], lag.max=8, exogen=datats2[,5],
          type="const")[["selection"]]
var3 <- VAR(datats2[,2:4], p=2, exogen=datats2[,5], type="const")
serial.test(var3, lags.pt=10, type="PT.asymptotic")
varfc3=forecast(var1,h=12,dumvar=matrix(rep(datats2[,5],12),12,1))
summary(varfc3)
autoplot(varfc3)
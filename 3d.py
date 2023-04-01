
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm,style
from mpl_toolkits.mplot3d import axes3d
from math import sqrt, pi,log, e
from datetime import datetime,timedelta
import pandas as pd
import requests
import scipy.stats as si
from scipy.stats import norm

style.use('dark_background')

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


### FUNÇÃO PARA CALCULAR BS
def bs(type,preco_spot, strike, dias_vcto = 30, taxa_selic = 0.1375, volat_stdv = 0.40, calendar_base = 365):
    # 'call' ou 'put'
    if type == 'CALL':
        type = 1
    else:
        type = -1

    # Initial parameters
    S = preco_spot
    K = strike
    T = dias_vcto/calendar_base
    r = taxa_selic
    sigma = volat_stdv
    q = 0 ### dividends

    # D1
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    # D2
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    # dfq = dividend Discount factor
    dfq = e ** -(q*T)
    # df = risk free Discount factor
    df = e ** -(r*T)

    # Premium
    if type == 1:
        premio = (S * norm.cdf(d1) - K * df * norm.cdf(d2))
    else:
        premio = (K * df * norm.cdf(-d2) - S * norm.cdf(-d1))
    
    # Delta
    if type == 1:
        delta = dfq * norm.cdf(d1)
    else:
        delta = dfq * (norm.cdf(d1) - 1)

    # Vega for 1% change in vol
    vega =  0.01 * S * dfq * norm.pdf(d1) * (T ** 0.5)

    # Theta for 1 day change
    if type == 1:
        theta = (1.0 / calendar_base) * (-0.5 * S * sigma * dfq * norm.pdf(d1) / (T ** 0.5) - r * K * df * norm.cdf(d2) + q*S*dfq*norm.cdf(d1))
    else:
        theta = (1.0 / calendar_base) * (-0.5 * S * sigma * dfq * norm.pdf(d1) / (T ** 0.5) + r * K * df * norm.cdf(-d2) - q*S*dfq*norm.cdf(-d1))

    # Gamma
    sigmaT = sigma * (T ** 0.5)
    gamma = dfq * norm.pdf(d1) / (S * sigmaT)

    return (premio,delta,vega,theta,gamma)

### FUNÇÃO PARA PEGAR TOKEN DE AUTENTICAÇÃO NA API
def get_token(email,senha):
    ## BODY PARA REQUISIÇÃO NA API
    body = {"email": email,"password": senha}
    
    ## CHAMADA NA API
    r = requests.post('https://api.oplab.com.br/v3/domain/users/authenticate',json=body).json()['access-token']
    return r

### FUNÇÃO PARA RETORNAR A SÉRIE HISTÓRICA
def getFechamentosPorData(token,symbol,data_inicio,data_fim,resolution="1d"):     
    ## HEADER DE AUTENTICAÇÃO
    header = {"Access-Token": token}
    
    ## CHAMADA NA API 
    dados = requests.get('https://api.oplab.com.br/v3/market/historical/{}/{}?from={}&to={}?smooth=true'.format(
    symbol, resolution, data_inicio.strftime("%Y%m%d%H%M"), data_fim.strftime("%Y%m%d%H%M")),
                        headers=header).json()['data']
    ## CONSTRUÇÃO DO DATAFRAME NO PANDAS
    fechamentos = []
    datas_list = []
    for i in dados:
        fechamentos.append(i['close'])
        datas_list.append(datetime.fromtimestamp(int(str(i['time'])[:10])))
    df = pd.DataFrame({'Adj Close': fechamentos}, index = datas_list)
    return df

### FUNÇÃO PARA RETORNAR A SÉRIE HISTÓRICA
def grade_dia(token,symbol,from_,to_,vctos = 1,call_put = 'PUT'):
    c = requests.get(
        'https://api.oplab.com.br/v3/market/historical/options/{}/{}/{}'.format(symbol,from_,to_),
        headers={"Access-Token": token}).json()
    lista_vcto_atual = []
    spot = []
    strikes = []
    vcto_list = []
    vols = []
    premio_list = []
    bs_list = []
    delta_list = []
    d1_list = []
    t_list = []
    gamma_list = []
    theta_list = []
    vega_list = []

    for i in c:
        mes_vcto = datetime.strptime(i['due_date'][:10], "%Y-%m-%d").month
        ano_vcto = datetime.strptime(i['due_date'][:10], "%Y-%m-%d").year
        maturity = (datetime.strptime(i['due_date'][:10], "%Y-%m-%d") - datetime.strptime(i['time'][:10], "%Y-%m-%d")).days
        if mes_vcto == (from_.month+vctos) and ano_vcto == from_.year and maturity > 10 and i['type'] == call_put:
            lista_vcto_atual.append(i)
    for j in lista_vcto_atual:
        # print(j)
        d = bs(call_put,j['spot']['price'], j['strike'], (datetime.strptime(j['due_date'][:10], "%Y-%m-%d") - datetime.strptime(j['time'][:10], "%Y-%m-%d")).days, 0.1375, j['volatility']/100)[1]
        t = bs(call_put,j['spot']['price'], j['strike'], (datetime.strptime(j['due_date'][:10], "%Y-%m-%d") - datetime.strptime(j['time'][:10], "%Y-%m-%d")).days, 0.1375, j['volatility']/100)[3]
        print(d)
        spot.append(j['spot']['price'])
        strikes.append(j['strike'])
        vcto_list.append((datetime.strptime(j['due_date'][:10], "%Y-%m-%d") - datetime.strptime(j['time'][:10], "%Y-%m-%d")).days)
        vols.append(j['volatility'])
        premio_list.append(j['premium'])
        bs_list.append(j['bs'])
        delta_list.append(j['delta'])
        d1_list.append(d)
        gamma_list.append(j['gamma'])
        theta_list.append(j['theta'])
        t_list.append(t)
        vega_list.append(j['vega'])

    df = pd.DataFrame({'s':spot,'k':strikes,'vctos':vcto_list,'vol':vols,'premio':premio_list,'bs':bs_list
                       ,'Delta':delta_list,'Gamma':gamma_list,'Theta':theta_list,'Vega':vega_list,'Delta1':d1_list,'Theta1':t_list})
    df = df.sort_values(by = 'k')

    return df

### FUNÇÃO PARA CALCULR A FUNÇÃO POLINOMIAL DA GRADE DO DIA REQUISITADO
def smile_do_dia(token,symbol,data_estudo,spot_price,lista_strikes,vctos = 1,range_moneyness=(0.10,1.8),graus_polin=3,call_put = 'PUT'):
    if isinstance(lista_strikes,list) == 0:
        print('Strikes deveria ser lista!')
        lista_strikes = []
    calculou = 0
    contador = 0
    while calculou == 0:
        contador += 1
        if contador > 5:
            break
        print(data_estudo)
        c = requests.get(
            'https://api.oplab.com.br/v3/market/historical/options/{}/{}/{}'.format(symbol, data_estudo.date(), data_estudo.date()),
            headers={"Access-Token": token}).json()
        lista_vcto_atual = []
        w = []
        strikes = []
        vols = []
        x = 0
        y = 0
        polynomio_list = []
        if len(c) < 5:
            data_estudo = data_estudo + timedelta(days=1)
            continue
        for i in c:
            moneyness = i['strike'] / i['spot']['price']
            mes_vcto = datetime.strptime(i['due_date'][:10], "%Y-%m-%d").month
            ano_vcto = datetime.strptime(i['due_date'][:10], "%Y-%m-%d").year
            maturity = (datetime.strptime(i['due_date'][:10], "%Y-%m-%d") - datetime.strptime(i['time'][:10], "%Y-%m-%d")).days
            if mes_vcto == (data_estudo.month+vctos) and ano_vcto == data_estudo.year and maturity > 10 and i['type'] == call_put and moneyness > range_moneyness[0] and moneyness < range_moneyness[1]:
                lista_vcto_atual.append(i)
        if len(lista_vcto_atual) < 3:
            data_estudo = data_estudo + timedelta(days=1)
            continue
        for j in lista_vcto_atual:
            strikes.append(j['strike'] / j['spot']['price'])
            vols.append(j['volatility'])
        if len(strikes) == 0 or len(vols) == 0:
            data_estudo = data_estudo + timedelta(days=1)
            continue
        df = pd.DataFrame({'s':strikes,'v':vols})
        df = df.sort_values(by = 's')
        x = list(df['s'])
        y = list(df['v'])
        if len(x) == 0 or len(y) == 0:
            data_estudo = data_estudo - timedelta(days=1)
            continue
        if len(x) > 0 and len(y) > 0:
            calculou = 1
            z = np.polyfit(x, y,graus_polin)
            polynomio_list = []
            for i in x:
                polynomio_list.append(np.poly1d(z)(i))
            if len(lista_strikes) == 0:
                continue
            else:
                for jj in lista_strikes:
                    mnnss = jj / spot_price
                    w.append(np.poly1d(z)(mnnss))
    return (polynomio_list,x,y,w)




### INSERIR EMAIL E SENHA --> get_token('seu@email.com','sua_senha')
try:
    token = get_token('','')
except:
    print('TOKEN ERRADO')
    exit()

### PARAMETROS INICIAIS DO ESTUDO
data_estudo = datetime(2023,1,16)


symbol = 'PETR4'
tipo = 'CALL'

spot = getFechamentosPorData(token,symbol,data_estudo,data_estudo)['Adj Close'][0]
spot_vol = getFechamentosPorData(token,symbol+'IVX',data_estudo,data_estudo)['Adj Close'][0]
vol = spot_vol/100




### CALCULAR GREGAS E CRIAR PLOTS
gregas = {'PREMIO':0,'DELTA':1,'VEGA':2,'THETA':3,'GAMA':4}

vctos = np.arange(1,45,1)
monness = np.arange(0.1,2,0.01)
monness_spot = np.arange(0.1*spot,2*spot,0.01*spot)

X,Y = np.meshgrid(vctos,monness)

fig = plt.figure(figsize=(9,8))

ax1 = fig.add_subplot(231, projection='3d')
ax2 = fig.add_subplot(232, projection='3d')
ax3 = fig.add_subplot(233, projection='3d')
ax4 = fig.add_subplot(234, projection='3d')
ax5 = fig.add_subplot(235, projection='3d')


# PREMIO
grega = 'PREMIO'

Z = bs(tipo,spot,Y*spot,X,0.019,vol)[gregas[grega]]

# Plot a 3D surface
ax1.plot_surface(X, Y, Z,cmap=cm.coolwarm)
ax1.set_xlabel('VENCIMENTO')
ax1.set_ylabel('MONEYNESS')
ax1.set_zlabel(grega)
ax1.set_title(grega)

# DELTA
grega = 'DELTA'

Z = bs(tipo,spot,Y*spot,X,0.019,vol)[gregas[grega]]

# Plot a 3D surface
ax2.plot_surface(X, Y, Z,cmap=cm.coolwarm)
ax2.set_xlabel('VENCIMENTO')
ax2.set_ylabel('MONEYNESS')
ax2.set_zlabel(grega)
ax2.set_title(grega)

# VEGA
grega = 'VEGA'

Z = bs(tipo,spot,Y*spot,X,0.019,vol)[gregas[grega]]

# Plot a 3D surface
ax3.plot_surface(X, Y, Z,cmap=cm.coolwarm)
ax3.set_xlabel('VENCIMENTO')
ax3.set_ylabel('MONEYNESS')
ax3.set_zlabel(grega)
ax3.set_title(grega)

# THETA
grega = 'THETA'

Z = bs(tipo,spot,Y*spot,X,0.019,vol)[gregas[grega]]

# Plot a 3D surface
ax4.plot_surface(X, Y, Z,cmap=cm.coolwarm)
ax4.set_xlabel('VENCIMENTO')
ax4.set_ylabel('MONEYNESS')
ax4.set_zlabel(grega)
ax4.set_title(grega)

# GAMA
grega = 'GAMA'

Z = bs(tipo,spot,Y*spot,X,0.019,vol)[gregas[grega]]

# Plot a 3D surface
ax5.plot_surface(X, Y, Z,cmap=cm.coolwarm)
ax5.set_xlabel('VENCIMENTO')
ax5.set_ylabel('MONEYNESS')
ax5.set_zlabel(grega)
ax5.set_title(grega)


### COLETAR DADOS DE SMILE E GRADE DO DIA
smile = smile_do_dia(token,symbol,data_estudo,40,[40],vctos = 1,range_moneyness=(0.80,1.1),graus_polin=3,call_put = tipo)
grade = grade_dia(token,symbol,data_estudo,data_estudo,1,tipo)
grade['mnnss'] = grade['k'] / grade['s'] - 1

### PLOTAR SMILE E STRIKES USADOS PARA CÁLCULO // PLOTAR GREGAS POR STRIKE
fig2, [bx1,bx2] = plt.subplots(2,1)
bx1.plot(smile[1],smile[2])
bx1.plot(smile[1],smile[0])
bx2.plot(grade['mnnss'],grade['Delta'])
bx2.plot(grade['mnnss'],grade['Vega'])
bx2.plot(grade['mnnss'],grade['Theta'])
bx2.plot(grade['mnnss'],grade['Delta1'])
bx2.plot(grade['mnnss'],grade['Theta1'])
bx1.set_title('Smile {} - {}'.format(symbol,data_estudo.date()))
bx1.legend(['Vértices Considerados','Polinômio'])
bx2.legend(['Delta','Vega','Theta','Delta1','Theta1'])


plt.show()

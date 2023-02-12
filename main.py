from random import random
import pandas as pd
import numpy as np
from numpy import sin, linspace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import Series


def diagrama_caja_1(notas, titulo,guarda,xlabe,ylabe):
    fig, ax = plt.subplots()
    notas.plot(kind = 'box', ax = ax)
    plt.xticks([])
    plt.title(titulo)
    plt.xlabel(xlabe)
    plt.ylabel(ylabe)
    fig.savefig(guarda)
    return ax
def diagrama_caja_m(notas, titulo,xlabe,ylabe):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(notas, patch_artist=True,
                    notch='True', vert=0)
    colors = ['#0000FF', '#00FF00',
              '#FFFF00', '#FF00FF']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)

    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)
    ax.set_yticklabels(['data_1', 'data_2',
                        'data_3', 'data_4'])

    plt.title(titulo)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.show()

    return ax

def diagrama_pastel(datos,nombres, titulo,guarda):
    fig, ax,junk = plt.pie(datos, labels=nombres, autopct="%0.1f %%")
    plt.title(titulo)
    plt.savefig(guarda)
    return ax

s_notas = pd.Series([5.7,8.5,9.1,5.5,8.2,9.0,10,7.0,7.7,9.9],index=["Juan","Jenifer","David","Pablo","Armando","Magdalena","Francesca","Rosmery","Vicente","Martin"])
print(s_notas)

diagrama_caja_1(s_notas, 'Distribución de notas','calificaciones.png',"alumnos","calificaciones")
plt.show()

manzanas = [20,10,25,30]
nombres = ["Ana","josss","Diana","Catalina"]
diagrama_pastel(manzanas,nombres,"jjs","cdabjb.png")
plt.show()

data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
notas = [data_1, data_2, data_3, data_4]
diagrama_caja_m(notas, "asjgdiuzxc","alumnos","calificaciones")
#,"fgggf.png",


def diagrama_lineas_ingresos_gastos(datos):
    fig, ax = plt.subplots()
    datos.plot(ax=ax)
    ax.set_ylim([0, max(datos.Ingresos.max(), datos.Gastos.max()) + 500])
    ax.set_ylabel('Dinero')
    ax.set_xlabel('MES')
    plt.title('Evolución de ingresos y gastos')
    return ax

def barra(titulo):#datos,nombres,guarda
    x = np.arange(7)
    y = np.random.randint(1, 20, 7)
    fig, ax = plt.subplots()
    ax.bar(x, y)
    plt.title(titulo)
    plt.show()


datos_EMPREZA=pd.read_csv("EMPRESA.csv")
print(datos_EMPREZA)

df_datos = pd.DataFrame(datos_EMPREZA).set_index('MES')
diagrama_lineas_ingresos_gastos(df_datos)
plt.show()

datos_titanic=pd.read_csv("train.csv")
print(datos_titanic)
datos_titanic.shape
x=datos_titanic["Age"].median()
datos_titanic["Age"].fillna(x,inplace=True)
y=datos_titanic["Pclass"].mean()
datos_titanic["Pclass"].fillna(y,inplace=True)
print(datos_titanic)
print(datos_titanic["Age"])
#diagrama_pastel(datos_titanic["Age"],edades, "edades de pasajeros","edades.png")
#plt.show()
diagrama_caja_1(datos_titanic["Survived"], " vivos y muertos",'muertos.png',"x","PERSONA")
plt.show()
diagrama_caja_1(datos_titanic["Pclass"], "clase de los pasajeros",'clase.png',"x","Clase")
plt.show()
print(datos_titanic.isnull().sum())
print(datos_titanic.describe())
barra("dcshoac")


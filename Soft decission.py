# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:57:50 2021

@author: PUSTIK
"""

import random as rd
from math import atanh, ceil, log, tanh
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import base64
import pandas as pd
from IPython.display import HTML

### (1) Inisialisasi Matriks H ###

a = int(input('Masukkan nilai kode siklik = '))
b = int(input('Masukkan panjang kode siklik = '))

# Permutasi dari kode dimana partisi paling ujung sama dengan ko-dimensi
partisi = np.zeros(b)               
for i in range (b):
    partisi[i] = rd.randint(2,a)
print('Cycle Structure')
print(partisi)
print()

# Panjang kode
nk = sum(partisi)                     
print('Panjang Kode')
print(nk)
print()

# Ko-dimensi kode LDPC
r = partisi[len(partisi)-1]         
print('Ko-dimensi Kode')
print(r)
print()

iter = 0
itermaks = 2

while iter <= itermaks:   
    # Baris pertama matriks H
    g = np.zeros(int(nk))                      
    for i in range (int(nk)):
        g[i] = rd.randint(0,1)
        
    # Matriks H
    dummy_g = []
    for i in g:
        dummy_g.append(i)
    dummy_g = np.array(dummy_g, dtype = np.int32)
    
    h = []
    for i in partisi:
        h.append(dummy_g[0:int(i)])
        dummy_g = dummy_g[int(i):len(dummy_g)]
    
    dummy_h = []
    for i in h:
        dummy_h.append(i)
    
    H = []
    for isi_H in range(int(r)):
        Hac = []
        Haci = []
        if isi_H != 0:
            for i in range(len(dummy_h)):
                roll = np.roll(dummy_h[i],1)
                Hac.append(roll)
                dummy_h[i] = roll
            for j in Hac:
                for k in j:
                    Haci.append(k)
            H.append(Haci)
        else:
            H.append(list(np.array(g, dtype = np.int32)))
    H = np.array(H)
    print('Matriks H')
    print(H)
    print()
    
    ### (2) Enkoding ###
    ### catatan : pastikan blok paling kanan matriks H sudah tak-singular ###
    
    # Hitung Matriks G
    Hk = []
    ind_awal = int((sum(partisi) - partisi[len(partisi)-1]))
    ind_akhir = int((ind_awal + partisi[len(partisi)-1]))
    Hk = H[:,ind_awal:ind_akhir]
    M = np.copy(Hk)
    determinan = np.linalg.det(M)
    
    if determinan != 0:
        iter = itermaks + 10
    else:
        iter = iter + 1
Gi = np.eye(int(nk)-int(r))
B = np.eye(len(M))
B = np.array(B, dtype = np.int32)
Mrref = np.column_stack((M,B))

# import numba
# parallel speeds up computation only over very large matrices
# numba.jit(nopython = True, parallel = True) 
# M is a mxn matrix binary matrix 
# all elements in M should be uint8

# Reduced Row Echelon Form (RREF) in GF(2) untuk invers Hk
# Thanks for popcornell for methode of gf2elim for find invers matrix
@jit(nopython = True)
def gf2elim(Mrref):
    m,n = Mrref.shape
      
    # Loop over the entire matrix
    i = 0
    j = 0

    while i < m and j < n:
        # Find value and index of largest element in remainder of column j
        k = np.argmax(Mrref[i:,j]) + i

        # Swap i-th and k-th rows
        # M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(Mrref[k])
        Mrref[k] = Mrref[i]
        Mrref[i] = temp
        
        # Save the right hand side of the pivot row
        aijn = Mrref[i,j:]

        # Make a copy otherwise M will be directly affected
        col = np.copy(Mrref[:,j])

        # Avoid xoring pivot row with itself
        col[i] = 0

        # This builds an matrix of bits to flip
        flip = np.outer(col,aijn)

        # Xor the right hand side of the pivot row with all the other rows
        Mrref[:,j:] = Mrref[:,j:]^flip

        i += 1
        j += 1

    return Mrref

C = gf2elim(Mrref)
o,p = gf2elim(Mrref).shape
Hk_inv = C[:,(p-o):p]

# Matriks G
G = []
isi_G = []
for i in range (len(partisi)-1):
    if i == 0:
        B1 = Hk_inv.dot((H[:,0:int(partisi[i])]))
    else:    
        ind1 = int(sum(partisi[0:i+1]) - int(partisi[i]))
        ind2 = int(sum(partisi[0:(i+1)]))
        B1 = Hk_inv.dot((H[:,ind1:ind2]))
    B2 = B1 % 2
    B3 = np.transpose(B2)
    isi_G.append(B3)
c = isi_G[0]
for i in range (1,len(isi_G)):
    c = np.concatenate((c,isi_G[i]))
G = np.column_stack((Gi,c))
# print('Matriks G')
# print(G)
# print()

# Generate pesan asli secara random
mes = np.zeros(int(nk)-int(r))
n1 = rd.randint(1,nk-r)
ind1 = list(np.random.permutation(np.arange(int(nk-r)))[:n1])
ind1 = np.array(ind1, dtype = np.int32)
for iterasi_pesan in range (len(ind1)):
    mes[ind1[iterasi_pesan]] = 1
# print('Pesan asli')
# print(mes)
# print()

# Hasil enkoding (pesan yang dikirim)
send = (mes.dot(G)) % 2
# print('Pesan yang dikirim')
# print(send)
# print()

# Bobot vektor error, t = 1
bataserror = ceil(2*nk/3)
inderror = []
for t in range (bataserror):
    t = t+1
    error = np.array(np.random.permutation(np.arange(int(nk)))[:t],
                     dtype = np.int32)
    inderror.append(error)

# Penambahan error ke pesan yang dikirim
receiverror = []
for i in range (len(inderror)):
    dummy_rcv = np.copy(send)
    for j in inderror[i]:
        dummy_rcv[int(j)] = (dummy_rcv[int(j)] + 1) % 2        
    receiverror.append(dummy_rcv)
receiverror = np.array(receiverror, dtype = float)        
# print('Pesan yang diterima')
# print(receiverror)
# print()

### (3) Inisialisasi Log-Likelihood Ratios Sum-Product ###
###                 Algorithm (LLR-SPA)                ###

BER = []
numerror = []
for s in range (len(receiverror)):
    # Algoritma LLR
    LLR = np.copy(receiverror[s])
    tl = s+1
    for i in range (int(nk)):
        if LLR[i] == 1:
            LLR[i] = log(int(tl)/(int(nk)-int(tl)))
        else:
            LLR[i] = log((int(nk)-int(tl))/int(tl))
    
    # Inisialisasi Gamma awal
    y = np.copy(receiverror[s])
    Ini_Gamma = np.zeros((int(nk),int(r)))
    Lambda = np.zeros((int(r),int(nk)))
    tg = s+1
    for i in range (int(nk)):
        for j in range (int(r)):
            if H[j,i] == 1:
                if y[i] == 1:
                    Ini_Gamma[i,j] = log(tg/(int(nk)-tg))
                else:
                    Ini_Gamma[i,j] = log((int(nk)-tg)/tg)
    # print('Gamma awal')
    # print(Ini_Gamma)
    # print()
    
    # Iterasi LLR-SPA
    iterasi = 0
    iterasimax = 15
    
    while iterasi <= iterasimax:

        #### (a) Left semi-iteration ####
    
        # A(k) = daftar semua tetangga dari check node ck
        Ak = []
        for i in range (len(H)):
            dummy_Ak = []
            for j in range (int(nk)):
                if H[i,j] == 1:
                    dummy_Ak.append(j)
            Ak.append(dummy_Ak)
        Ak = np.array(Ak, dtype = np.int32)
    
        # Nilai Lambda terupdate
        for e in range (len(Ak)):
            Aki = np.copy(Ak[e])
            for i in Aki:
                lokasi = np.where(Aki == i)
                isi_Aki = np.delete(Aki,lokasi)
                isi_pi = 1
                for j in isi_Aki:
                    isi_pi *= tanh(1/2*(Ini_Gamma[j,e]))
                Lambda[e,i] = 2*atanh(isi_pi)
        # print('Lambda terbaru')
        # print(Lambda)
        # print()
    
        #### (b) Right semi-iteration ####
    
        # B(i) = daftar semua tetangga dari variabel node vi
        Bi = []
        HH = np.transpose(H)
        for i in range (int(nk)):
            dummy_Bi = []
            for j in range (len(H)):
                if HH[i,j] == 1:
                    dummy_Bi.append(j)
            Bi.append(dummy_Bi)
    
        # Nilai Gamma terupdate dan Gamma_i untuk keputusan
        Gamma = np.zeros((int(nk),int(r)))
        Gammadec = np.zeros(int(nk))
        for e in range (len(Bi)):
            Bik = np.copy(Bi[e])
            if len(Bik) != 0:
                for i in Bik:
                    lokasi1 = np.where(Bik == i)
                    isi_Bik = np.delete(Bik,lokasi1)
                    isi_sigma = 0
                    if len(isi_Bik) != 0:
                        for j in isi_Bik:
                            isi_sigma += Lambda[j,e]
                            Gamma[e,i] = LLR[e] + isi_sigma
                    else:
                        Gamma[e,i] = LLR[e]
                isi_sigmabaru = 0
                for k in Bik:
                    isi_sigmabaru += Lambda[k,e]
                    Gammadec[e] = LLR[e] + isi_sigmabaru
        # print('Gamma terupdate')
        # print(Gamma)
        # print()
        # print('Gamma_i untuk keputusan')
        # print(Gammadec)
        # print()

        ### (4) Decission ###

        koreksi = np.zeros(int(nk))
        for i in range (int(nk)):
            if Gammadec[i] >= 0:
                koreksi[i] = 0
            else:
                koreksi[i] = 1
        # print('receive hasil decission')
        # print(koreksi)
        # print()
        
        syndrome = sum(H*koreksi) % 2
        if list(koreksi) == list(send):
            iterasi = iterasimax + 10
        else:
            iterasi = iterasi + 1
            ini_Gamma = np.copy(Gamma)
    receivecorr = np.copy(koreksi)
    # print('receive hasil koreksi')
    # print(receivecorr)
    # print()
    
    # Hitung Bit Error Rate (BER)
    selv = (send-receivecorr) % 2

    selind = []
    for i in range (len(selv)):
        if selv[i] == 1:
            selind.append(i)

    isi_ber = len(selind)/int(nk)
    BER.append(isi_ber)

    # Banyaknya nomor error
    numerror.append(s+1)
    
    def create_download_link(df, title = 'Download CSV file',
                             filename = 'soft decission'):
        csv = df.to_csv()
        b64 = base64.b64encode(csv.encode())
        payload = b64.decode()
        html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(payload=payload,title=title,filename=filename)
        return HTML(html)

    df = pd.DataFrame(data = {'Nama' : ['Nilai LLR','Gamma awal',
                                        'Lambda terupdate','Gamma terupdate',
                                        'Gamma untuk keputusan',
                                        'Hasil decision','Hasil koreksi'],
                              'Hasil' : [LLR,Ini_Gamma,Lambda,Gamma,Gammadec,
                                         koreksi,receivecorr]})
create_download_link(df)
    
ula = 0
ulamax = 500

while ula <= ulamax:
    BERF = []
    for i in range (len(BER)):
        if i > 0:
            if BER[i-1] >= BER[i]:
                BERF.append(max(BER))
            else:
                BERF.append(BER[i])
        else:
            BERF.append(BER[0])
    
    for j in range (len(BERF)):
        if j > 0:
            if BER[j-1] >= BER[j]:
                ula = ula + 1
                BER = np.copy(BERF)
            else:
                continue

# Plot BER terhadap banyaknya error yang ditambahkan
Sumbu_x = np.copy(numerror)
Sumbu_y = np.copy(BERF)
plt.xlabel('Number of errors')
plt.ylabel('Bit Error Rate')
plt.title('Bit Error Rate vs Error added')
plt.plot(Sumbu_x, Sumbu_y, color = 'blue', linewidth = 2, marker = 'o',
          markerfacecolor = 'orange', markersize = 10)
plt.show()
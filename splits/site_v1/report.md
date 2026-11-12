# Split per sito (site_v1)

- val: ['fair', 'nus'] - test: ['georgiatech']
- take: train 1138, val 68, test 67
- criteri: classe valutata se >= 8 take in val/test; servono >= 2 siti di train con >= 5 take della classe
- accuratezza di 'classe piu' frequente del sito' sul train (in-sample): 0.784
- problemi di copertura: nessuno
- mappa sito -> fold (split.json): {'uniandes': 0, 'unc': 1, 'sfu': 2, 'cmu': 3, 'iiith': 3, 'minnesota': 3, 'indiana': 4, 'upenn': 4, 'utokyo': 4}

## train: sito x classe

```
classe     Basketball  Bike repair  Bouldering  Cooking  Dance  Health  Soccer   All
site                                                                                
cmu                 0           10           0        0      0       0       0    10
iiith               0            0           0       49      0       0      62   111
indiana             0           11           0        9      0       0       0    20
minnesota           0            0           0        1      0       0       0     1
sfu               102            0           0       18      0      10       0   130
unc               244            0           0        0      0       0       0   244
uniandes          122            0           2        9    362       0       0   495
upenn               0            0           0       18     90       0       0   108
utokyo              0            0           0        1      0      10       8    19
All               468           21           2      105    452      20      70  1138
```

## val: sito x classe

```
classe  Bike repair  Cooking  Health  All
site                                     
fair             21        3       0   24
nus               0        8      36   44
All              21       11      36   68
```

## test: sito x classe

```
classe       Bike repair  Cooking  Health  All
site                                          
georgiatech           39        7      21   67
All                   39        7      21   67
```

## copertura per classe

```
             train_take  train_siti  val_take  test_take
classe                                                  
Basketball          468           3         0          0
Dance               452           2         0          0
Soccer               70           2         0          0
Cooking             105           5        11          7
Bouldering            2           0         0          0
Health               20           2        36         21
Bike repair          21           2        21         39
```

## fold del train (per sito)

```
                      siti  take classi_senza_train
fold                                               
0                 uniandes   495         Bouldering
1                      unc   244                  -
2                      sfu   130                  -
3      cmu,iiith,minnesota   122                  -
4     indiana,upenn,utokyo   147                  -
```

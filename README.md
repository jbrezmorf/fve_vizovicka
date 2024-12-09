# FVE Vizovická - modely

Postup:
1. Model one year of clear sky production, use CET.
2. Bar plot individual hours on X, month + value on Y, use different colors.
3. Plot individual days shifted in X  as dots
4. Nearl perfect agreement between irradiation models.
6. Get spot prices over last two years.


==
5. agreement for production for Kadlec and JB
6. Use real weather for 2023 and 2024
7. Use spot prices

4. Model and compare actual total production in kWh for indivisual hours.
Kadlec:
TB: 9 vychod, 9 zapad, surf 39.9m2

JB: 18 vychod 18 zapad, surf 79.8m2
AM - pohlcené záření na západ (kWh/m2); Budovy 02-Oblast modulu Západ: Globální záření na modul 
AU - pohlcené záření na východ (kWh/m2); Budovy 02-Oblast modulu Východ: Globální záření na modul 
BP - výkon FVE západ (kWh); Střídač 1 - MPP 1 - do  Budovy 02-Oblast modulu Východ & Budovy 02-Oblast modulu Západ: FV globální záření 
CJ - výkon FVE východ (kWh); Střídač 1 - MPP 2 - do  Budovy 02-Oblast modulu Východ & Budovy 02-Oblast modulu Západ: FV globální záření 

Vybíjení baterky:
DO (vybíjení) + DQ (ztráty: AC/DC ztráty) + DS (ztráty: kabely) = DL (dodávka FV)

Dobíjení baterky:
DM (energie z FV, vstup měniče) + DN (dobíjení) + DP (ztráty: odchylka od jmenovitého napětí) + DQ (ztráty: AC/DC ztráty) + DS (ztráty: kabely) = DL (dodávka FV)

celkem:
DO + DM + DN + DP + DQ + DS = DL

DM  = AZ (FV jmenovitá energie) + BA (chování za nízké intenzity) + BB (specifické dílčí stínění modulu) + BC (odchylka od jmenovité teploty)
AZ = AV (Globální záření) + AY (STC ztráty konverze)
     11.346

AV = (irrad_eff_e * surface_e + irrad_eff_w * surface_w)
irrad_eff_* = irrad_na_modulu + korekce spektra (-1%)+ odraz od země (+0.1%) + odraz na modulu (-7%)

irrad_w: 0.143/m2 = 11.411 kWh
irrad_e: 0.252/m2= 20.110 kWh

5. 

4. Download weather, plot production affecte dby weather (need more realistic PV model including diffused light)
5. Compare to model from Kadlec. 
6. Get spot prices over last two years.
7. Optimize angle to maximize revenue if only can sell or buy. 
   Real revenues would be higher since any autonomous consumption of the produced energy is net income.
   This is in max. contrast to the optimisation without selling.
   
   
8. Rule based  model of consumption. Const profile over day + linear corralation of heating to temperature.
9. Optimized consumption: control variables: 0/1 heat pump, 0/1 FV, 0-1 grid, střídač výkon +/-
   

Self shading:
https://pvlib-python.readthedocs.io/en/latest/gallery/shading/plot_passias_diffuse_shading.html#sphx-glr-gallery-shading-plot-passias-diffuse-shading-py


   Střídač umí přepínat směry a nastavovat výkon. (možné technické řešení)
    
   predictions:
   - heat consumption, building heat model, including (heat pump on/off control), function of weather
   - uncontroled consumption (stochastic, empirical, function of: time, weather)
   - FV production, function of weather
   - stock prices (stochastic, emprical, function of weather)
  
   Having all necessary models one can use a genetic algorithm approach, switching every control every half an hour 
   or so. Detailed shorter period control to/from battery, short period control should only turn grid off if \
   SPOT prices change every about 5-15 minutes. So for 15 min. intervals, predicting 4 15min intervals 4 half hour, 4 hour, 4 2 hour 4 4 hour = 16 + 8 + 4 + 2 + 1= 31h
   20 intervalů, 3 binární proměnné * 8 úrovní střídače = 64^20 stavů !!. Alternativně lze aplikovat spojité řízení se spínáním přes hysterezi řídících proměnných. 
   Fakticky se tak pouze modifikuje loos function, která není hladká případně není spojitáPři penalizování Pro každou volbu genetického algoritmu by se pro každý interval nastavila

## Consumption algorithm

Sources:
grid (buy)
FV
battery

Drains:
Heat pump
Direct Heating (not installed, reduction ventil may be necessary)
Battery (charge)
Grid (sell)
Uncontroled

Algorithm must determine for each drain when it is activated and from which source it takes power.

Whitout heat pump and speculative buying:
Uncontroled consumption source: 1. from FV 2. from battery 3. from grid
FV drain: 1. consumption 2. battery charging 3. gird sell, but only if it gives at least some revenue

Prediction of FV and consumption -> prediction of energy balance 

Battery states:
NO battery, Battery BELLOW/MIDDLE/OVER expected negative energy balance, FULL
indicator: how long can battery absorb predicted energy balance?


Heating priorities: 
zero (no heat capacity), low (have some capacity but temperatures at max.), mid (some temperatures bellow min), high (all temp. below min)


Only if has free storage capacity (not overheated). 
- always if BATTERY FULL and have FV production (kolísání doplní baterka)
- 
always if good FV production  
- if price is under threshold
- if battery s OVER or FULL
- if Free battery capacity (prefere discharge
- or FV production
- or price under threshold

Temperature or WW under minimum temperature. 
Start with grid source.


Speculative charging on:
We distinguish regimes according to local energy balance:

## No FV production.


### No battery, Temperature or WW under min level
Start heat pump. Buy at any cost from grid. (should be avoided)

### Battery at or bellow expected 6 hours Uncontroled consupmtion (nearest peak)No battery, Temperature and WW OK.
Buy what for Uncontroled. Heat pump. stoped.

### Battery, over expected consumptin.

### Battery full


Heat pump / direct heating
Direct heating should only be used as the last resort 

### Use production or battery for any uncontroled consumption.
### Charge battery from FVE whenever 
### Battery charging form grid:


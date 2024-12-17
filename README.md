# FVE Vizovická - modely

Model summary:
- FVE production model using pvlib, when compared to Kadlec model using its GHI and DHI the 
  DC production was overestimated; could not determine the reason, but able to construct a 
  cubic model dependent on model_dc and air temperature to get agreement with Kadlec model up to about +/-7% 
  and matching mean production.
- the weather for years up to 2023 for any given location is downloaded from pvGIS
- spot prices (EUR/MWh) for each hour are extracted from OTE reports, having years from 2018
- since we want to exclude years affected by covid and war related energy violated prices we parform optimization for 
  2018, 2019 and 2023
- the consumption model is taken from Kadlec model with a correction that the portion attributed to the heating 
  is correlated to the the negetive part of (air temp - 10).
- A simple selling logic is applied:
  1. for each day we determine reminded energy and assing it to the hours with max product price*production
  2. we go sequentialy over the day hours trying to sell planed value use reminded energy to charge battery and discharge for
     for consumption.
  3. What could not be coverted by battery is back propagated to buy/sell model
  4. buy/sell accounts for distribution fees given by "BezDodavatele.cz" about 4Kč dayly and 450 for MWh buyed or sell.

- A simple optimization was performed first computing verians for the 'east' pannels resulting to 
  'optimal' value about 110 deg azimuth, 50 deg tilt; Then the west pannels were changed resulting to 
- 'optimal' on the east bound of the azimuth range. Orienting all pannels to the east could be the best choice after all.
  But that could be result of poor sell model.
- Dayly curves of production and irradiation especially for the east orientations exhibit a morning peaks. These seems to be already in the 
  pvGIS data. Not sure if these are real. That could also affect the optimization results.

TODO:
- tray to make optimization more robust in order to optimize for different years, models
  1. parametrize optimization be model setting
  2. three stage optimization : 
        1. compute for all pannels in same direction, tilt with large step
        2. determine best 10 cases continue with a genetic optimization
        3. loss: a * revenue + b * dispersion ; sensitivity of revenue to weather and price
        2. find optimal sell model for each year
        3. find optimal battery management model for each year
- empirical sell model including optimisation of consumption, price*consumption coverd should be treated like revenue

- better battery management model:
  - collect all days togehter with starting battery state for different pannel setting and different years
  - use indeger control programming to optimize
  - train NN model to predict sell for next hour using production and price prediciton (noised) and current state 
notes:
podmínky prodeje / připojení
- článek na TZB info: https://oze.tzb-info.cz/fotovoltaika/26135-jak-se-ucinne-branit-proti-fakturam-za-pretoky-u-mikrozdroju
- řada technických podmínek zajišťujících kvalitu výroby (napětí, frekvence, fáze) to by měl řešit kvalitní střídač
- nutno mít od distributora "rezervovaný výkon" za překročení se platí pokuta cca 1713/ (kWměsíc) = 2.38 Kč/kWh
- pro zdroje na 3 fázích se pokuta počítá pro překročení rezervovaného výkonu nad 300W 
- překročená prováděno měsíčně (? platí i nově nebo je to po hodině)
Ověřit schopnosti střídače: 
- zajistit kvalitu výroby
- asymetrický sřídač, adaptivně krmí fáze podle potřeby
- připojení na jednotlivé fáze je asi až v bytových rozvaděčích a otázka je jak je vyvážené ??

Kadles 3.1. 10~9:30 
Kaclec eff irrad 11.346/79.8 = 0.142 kW/m2;
fvmodel (0.2kW/m2 for both east and west !!, one is ok other should be about 0.08kW/m2)

Kadlec total 11kWh irrad, -> -8.8 conv loss -> 2.2kWh (22% efficiency); cca 60kWh modul ->  
fve predicts 4.2 kWhfor just single array!!
from pannel performance, using just linear scaling it should be about 3.2kWh 

Postup:


6. Use real weather for 2023 and 2024
=======================================================
Kadlec tables:
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

=======================================================

   

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


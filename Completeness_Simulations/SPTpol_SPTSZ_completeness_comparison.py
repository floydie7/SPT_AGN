"""
SPTpol_SPTSZ_completeness_comparison.py
Author: Benjamin Floyd

Generates a comparison plot of the two survey completeness curves.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# Filenames
sptsz_filename = 'Data/Comp_Sim/Results/SPT_I2_results_gaussian_fwhm202_corr011_mag02.json'
sptpol_filename = 'Data/Comp_Sim/SPTpol/Results/SPTpol_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'

# Read in the SPT-SZ completeness results
with open(sptsz_filename, 'r') as f:
    sptsz_results = json.load(f)

# Read in the SPTpol completeness results
with open(sptpol_filename, 'r') as f:
    sptpol_results = json.load(f)

# Extract the magnitude bins entry from the dictionaries with a 0.25 shift to center the magnitude bin on the data point
mag_bins = np.array(sptsz_results.pop('magnitude_bins')[:-1]) + 0.25
del sptpol_results['magnitude_bins']

# Remove SPT-CLJ2341-5640 (bad photometry, large star)
del sptpol_results['SPT-CLJ2341-5640']

# Get the list of values
sptsz_values = list(sptsz_results.values())
sptpol_values = list(sptpol_results.values())

# Compute the median curves
sptsz_med_curve = np.median(sptsz_values, axis=0)
sptpol_med_curve = np.median(sptpol_values, axis=0)

# Find the curve envelopes
sptsz_min_curve = np.min(sptsz_values, axis=0)
sptsz_max_curve = np.max(sptsz_values, axis=0)
sptpol_min_curve = np.min(sptpol_values, axis=0)
sptpol_max_curve = np.max(sptpol_values, axis=0)

# Make plot
fig, ax = plt.subplots()
ax.axvline(17.46, color='k', linestyle='--', alpha=0.2)
ax.axhline(0.8, color='k', linestyle='--', alpha=0.2)
ax.plot(mag_bins, sptsz_med_curve, color='C0', label='SPT-SZ')
ax.plot(mag_bins, sptpol_med_curve, color='C1', label='SPTpol')
ax.fill_between(mag_bins, sptsz_min_curve, sptsz_max_curve, facecolor='C0', alpha=0.4)
ax.fill_between(mag_bins, sptpol_min_curve, sptpol_max_curve, facecolor='C1', alpha=0.4)
ax.legend()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set(title=r'Completeness Simulation Comparison for IRAC 4.5 $\mu$m', xlabel='Vega Magnitude', ylabel='Recovery Rate')
fig.savefig('Data/Comp_Sim/SPTpol/Plots/SPTpol_SPTSZ_comparison_no_SPT2341-5640_limit_line_corrected.pdf', format='pdf')
# plt.show()

# # SPT-CLJ2355-5156 Comparison
# spt2355_file = 'Data/Comp_Sim/SPTpol/Results/SPT-CLJ2355-5156_I2_results_targeted_irac_data_sptsz_ssdf_irac_data_sptpol.json'
# with open(spt2355_file, 'r') as f:
#     spt2355_results = json.load(f)
#
# targeted_curve = spt2355_results['targeted_sptsz']
# ssdf_curve = spt2355_results['ssdf_sptpol']
#
# fig, ax = plt.subplots()
# ax.plot(mag_bins, targeted_curve, label='Targeted Obs (Cycle 11)')
# ax.plot(mag_bins, ssdf_curve, label='SSDF Cutout (Cycle 8)')
# ax.legend()
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax.set(title=r'Completeness Simulation of SPT-CLJ2355-5156 for IRAC 4.5 $\mu$m',
#        xlabel='Vega Magnitude', ylabel='Recovery Rate')
# fig.savefig('Data/Comp_Sim/SPTpol/Plots/SPT-CLJ2355-5156_comparison.pdf', format='pdf')

# Compare the common clusters
# First, as the SPT-SZ results are stored according to their observation ID rather than their official ID we need a
# look-up database. This can be made by running the first two stages of the selection pipeline then running
# `sptsz_off_to_obs_ids = {Bocquet['SPT_ID'][info['SPT_cat_idx']]: obs_id for obs_id, info in selector._catalog_dictionary.items()}`
# The result is here for convenience. Format is {official_id: observation_id}
sptsz_id_lookup = {'SPT-CLJ0000-4356': 'SPT-CLJ0000-4356',
                   'SPT-CLJ0000-5748': 'SPT-CLJ0000-5748',
                   'SPT-CLJ0001-4024': 'SPT-CLJ0001-4024',
                   'SPT-CLJ0001-5440': 'SPT-CLJ0001-5439',
                   'SPT-CLJ0002-5557': 'SPT-CLJ0002-5557',
                   'SPT-CLJ0011-4614': 'SPT-CLJ0011-4613',
                   'SPT-CLJ0014-4036': 'SPT-CLJ0014-4036',
                   'SPT-CLJ0015-6000': 'SPT-CLJ0015-5959',
                   'SPT-CLJ0019-5527': 'SPT-CLJ0019-5526',
                   'SPT-CLJ0021-4902': 'SPT-CLJ0021-4902',
                   'SPT-CLJ0027-4742': 'SPT-CLJ0027-4742',
                   'SPT-CLJ0037-5047': 'SPT-CLJ0037-5047',
                   'SPT-CLJ0043-4843': 'SPT-CLJ0043-4843',
                   'SPT-CLJ0044-4037': 'SPT-CLJ0044-4037',
                   'SPT-CLJ0048-5244': 'SPT-CLJ0048-5244',
                   'SPT-CLJ0048-6416': 'SPT-CLJ0048-6415',
                   'SPT-CLJ0049-5315': 'SPT-CLJ0049-5314',
                   'SPT-CLJ0058-6145': 'SPT-CLJ0058-6145',
                   'SPT-CLJ0100-5359': 'SPT-CLJ0100-5358',
                   'SPT-CLJ0102-4603': 'SPT-CLJ0102-4603',
                   'SPT-CLJ0103-4250': 'SPT-CLJ0103-4249',
                   'SPT-CLJ0104-4351': 'SPT-CLJ0104-4351',
                   'SPT-CLJ0108-4659': 'SPT-CLJ0108-4659',
                   'SPT-CLJ0117-6032': 'SPT-CLJ0117-6032',
                   'SPT-CLJ0118-5156': 'SPT-CLJ0118-5156',
                   'SPT-CLJ0123-4821': 'SPT-CLJ0123-4821',
                   'SPT-CLJ0131-5604': 'SPT-CLJ0131-5604',
                   'SPT-CLJ0131-5921': 'SPT-CLJ0131-5921',
                   'SPT-CLJ0139-5204': 'SPT-CLJ0139-5204',
                   'SPT-CLJ0140-4833': 'SPT-CLJ0140-4833',
                   'SPT-CLJ0142-5032': 'SPT-CLJ0142-5032',
                   'SPT-CLJ0147-5622': 'SPT-CLJ0147-5622',
                   'SPT-CLJ0148-4518': 'SPT-CLJ0148-4518',
                   'SPT-CLJ0148-4700': 'SPT-CLJ0148-4659',
                   'SPT-CLJ0151-5954': 'SPT-CLJ0151-5954',
                   'SPT-CLJ0152-5303': 'SPT-CLJ0152-5303',
                   'SPT-CLJ0154-4824': 'SPT-CLJ0154-4824',
                   'SPT-CLJ0156-5541': 'SPT-CLJ0156-5541',
                   'SPT-CLJ0157-6442': 'SPT-CLJ0157-6442',
                   'SPT-CLJ0202-5401': 'SPT-CLJ0202-5401',
                   'SPT-CLJ0205-5829': 'SPT-CLJ0205-5829',
                   'SPT-CLJ0212-4657': 'SPT-CLJ0212-4656',
                   'SPT-CLJ0216-4219': 'SPT-CLJ0216-4219',
                   'SPT-CLJ0216-4830': 'SPT-CLJ0216-4830',
                   'SPT-CLJ0216-6409': 'SPT-CLJ0216-6409',
                   'SPT-CLJ0217-4310': 'SPT-CLJ0217-4310',
                   'SPT-CLJ0218-4233': 'SPT-CLJ0218-4232',
                   'SPT-CLJ0218-4315': 'SPT-CLJ0218-4315',
                   'SPT-CLJ0219-4934': 'SPT-CLJ0219-4934',
                   'SPT-CLJ0221-4446': 'SPT-CLJ0221-4446',
                   'SPT-CLJ0230-6028': 'SPT-CLJ0230-6028',
                   'SPT-CLJ0231-5403': 'SPT-CLJ0231-5403',
                   'SPT-CLJ0233-5819': 'SPT-CLJ0232-5819',
                   'SPT-CLJ0238-4904': 'SPT-CLJ0238-4904',
                   'SPT-CLJ0242-4944': 'SPT-CLJ0242-4944',
                   'SPT-CLJ0243-5930': 'SPT-CLJ0243-5930',
                   'SPT-CLJ0256-5617': 'SPT-CLJ0256-5617',
                   'SPT-CLJ0258-5355': 'SPT-CLJ0258-5355',
                   'SPT-CLJ0259-4556': 'SPT-CLJ0259-4555',
                   'SPT-CLJ0310-4647': 'SPT-CLJ0310-4646',
                   'SPT-CLJ0324-6236': 'SPT-CLJ0324-6236',
                   'SPT-CLJ0334-4645': 'SPT-CLJ0334-4645',
                   'SPT-CLJ0337-6300': 'SPT-CLJ0337-6300',
                   'SPT-CLJ0339-4545': 'SPT-CLJ0339-4544',
                   'SPT-CLJ0341-5731': 'SPT-CLJ0341-5731',
                   'SPT-CLJ0341-6143': 'SPT-CLJ0341-6142',
                   'SPT-CLJ0344-5452': 'SPT-CLJ0344-5452',
                   'SPT-CLJ0345-6419': 'SPT-CLJ0345-6419',
                   'SPT-CLJ0346-5839': 'SPT-CLJ0346-5839',
                   'SPT-CLJ0351-4109': 'SPT-CLJ0350-4109',
                   'SPT-CLJ0350-4620': 'SPT-CLJ0350-4619',
                   'SPT-CLJ0352-5647': 'SPT-CLJ0352-5648',
                   'SPT-CLJ0354-4058': 'SPT-CLJ0354-4058',
                   'SPT-CLJ0354-5151': 'SPT-CLJ0354-5151',
                   'SPT-CLJ0354-6032': 'SPT-CLJ0354-6032',
                   'SPT-CLJ0356-5337': 'SPT-CLJ0356-5337',
                   'SPT-CLJ0357-4521': 'SPT-CLJ0357-4521',
                   'SPT-CLJ0359-5218': 'SPT-CLJ0359-5218',
                   'SPT-CLJ0402-6130': 'SPT-CLJ0402-6130',
                   'SPT-CLJ0403-5719': 'SPT-CLJ0403-5719',
                   'SPT-CLJ0404-4418': 'SPT-CLJ0404-4418',
                   'SPT-CLJ0405-4648': 'SPT-CLJ0405-4648',
                   'SPT-CLJ0406-5455': 'SPT-CLJ0406-5455',
                   'SPT-CLJ0408-4456': 'SPT-CLJ0408-4456',
                   'SPT-CLJ0412-5106': 'SPT-CLJ0412-5106',
                   'SPT-CLJ0417-4427': 'SPT-CLJ0417-4427',
                   'SPT-CLJ0417-4748': 'SPT-CLJ0417-4748',
                   'SPT-CLJ0418-4552': 'SPT-CLJ0418-4552',
                   'SPT-CLJ0421-4845': 'SPT-CLJ0421-4845',
                   'SPT-CLJ0422-5140': 'SPT-CLJ0422-5140',
                   'SPT-CLJ0426-5416': 'SPT-CLJ0426-5416',
                   'SPT-CLJ0428-6049': 'SPT-CLJ0428-6049',
                   'SPT-CLJ0429-5233': 'SPT-CLJ0429-5233',
                   'SPT-CLJ0432-6150': 'SPT-CLJ0432-6150',
                   'SPT-CLJ0433-5630': 'SPT-CLJ0433-5630',
                   'SPT-CLJ0441-4855': 'SPT-CLJ0441-4854',
                   'SPT-CLJ0442-6138': 'SPT-CLJ0442-6138',
                   'SPT-CLJ0444-5603': 'SPT-CLJ0444-5603',
                   'SPT-CLJ0445-4230': 'SPT-CLJ0445-4230',
                   'SPT-CLJ0446-4606': 'SPT-CLJ0446-4605',
                   'SPT-CLJ0446-5849': 'SPT-CLJ0446-5849',
                   'SPT-CLJ0447-5055': 'SPT-CLJ0447-5054',
                   'SPT-CLJ0449-4901': 'SPT-CLJ0449-4901',
                   'SPT-CLJ0451-4952': 'SPT-CLJ0451-4952',
                   'SPT-CLJ0454-4211': 'SPT-CLJ0454-4210',
                   'SPT-CLJ0456-4906': 'SPT-CLJ0456-4906',
                   'SPT-CLJ0456-5116': 'SPT-CLJ0456-5116',
                   'SPT-CLJ0458-5741': 'SPT-CLJ0458-5741',
                   'SPT-CLJ0459-4947': 'SPT-CLJ0459-4947',
                   'SPT-CLJ0509-5342': 'SPT-CLJ0509-5342',
                   'SPT-CLJ0511-5154': 'SPT-CLJ0512-5154',
                   'SPT-CLJ0516-5430': 'SPT-CLJ0517-5430',
                   'SPT-CLJ0517-6119': 'SPT-CLJ0517-6119',
                   'SPT-CLJ0521-5104': 'SPT-CLJ0521-5105',
                   'SPT-CLJ0528-5300': 'SPT-CLJ0528-5300',
                   'SPT-CLJ0529-4138': 'SPT-CLJ0529-4138',
                   'SPT-CLJ0529-6051': 'SPT-CLJ0529-6051',
                   'SPT-CLJ0530-4139': 'SPT-CLJ0530-4138',
                   'SPT-CLJ0533-5005': 'SPT-CLJ0534-5005',
                   'SPT-CLJ0534-5937': 'SPT-CLJ0534-5938',
                   'SPT-CLJ0535-4801': 'SPT-CLJ0535-4801',
                   'SPT-CLJ0535-5956': 'SPT-CLJ0535-5956',
                   'SPT-CLJ0536-6109': 'SPT-CLJ0536-6109',
                   'SPT-CLJ0537-6504': 'SPT-CLJ0537-6504',
                   'SPT-CLJ0540-5744': 'SPT-CLJ0540-5745',
                   'SPT-CLJ0543-4250': 'SPT-CLJ0543-4250',
                   'SPT-CLJ0543-6219': 'SPT-CLJ0543-6219',
                   'SPT-CLJ0546-5345': 'SPT-CLJ0547-5345',
                   'SPT-CLJ0549-6205': 'SPT-CLJ0549-6205',
                   'SPT-CLJ0551-4339': 'SPT-CLJ0551-4339',
                   'SPT-CLJ0551-5709': 'SPT-CLJ0552-5709',
                   'SPT-CLJ0555-6406': 'SPT-CLJ0555-6406',
                   'SPT-CLJ0556-5403': 'SPT-CLJ0556-5403',
                   'SPT-CLJ0557-4113': 'SPT-CLJ0557-4113',
                   'SPT-CLJ0557-5116': 'SPT-CLJ0557-5116',
                   'SPT-CLJ0559-5249': 'SPT-CLJ0600-5249',
                   'SPT-CLJ0607-4448': 'SPT-CLJ0607-4448',
                   'SPT-CLJ0611-4724': 'SPT-CLJ0611-4724',
                   'SPT-CLJ0611-5938': 'SPT-CLJ0611-5938',
                   'SPT-CLJ0612-4317': 'SPT-CLJ0612-4317',
                   'SPT-CLJ0613-5627': 'SPT-CLJ0613-5627',
                   'SPT-CLJ0615-5746': 'SPT-CLJ0615-5746',
                   'SPT-CLJ0616-4407': 'SPT-CLJ0616-4407',
                   'SPT-CLJ0617-5507': 'SPT-CLJ0617-5508',
                   'SPT-CLJ0619-5802': 'SPT-CLJ0619-5802',
                   'SPT-CLJ0622-4645': 'SPT-CLJ0622-4645',
                   'SPT-CLJ0625-4330': 'SPT-CLJ0625-4329',
                   'SPT-CLJ0626-4446': 'SPT-CLJ0626-4446',
                   'SPT-CLJ0640-5113': 'SPT-CLJ0640-5113',
                   'SPT-CLJ0641-4733': 'SPT-CLJ0641-4733',
                   'SPT-CLJ0641-5950': 'SPT-CLJ0641-5950',
                   'SPT-CLJ0643-4535': 'SPT-CLJ0643-4535',
                   'SPT-CLJ0648-4622': 'SPT-CLJ0648-4622',
                   'SPT-CLJ0649-4510': 'SPT-CLJ0649-4510',
                   'SPT-CLJ0655-5234': 'SPT-CLJ0655-5234',
                   'SPT-CLJ2011-5228': 'SPT-CLJ2011-5228',
                   'SPT-CLJ2017-6258': 'SPT-CLJ2018-6259',
                   'SPT-CLJ2019-5642': 'SPT-CLJ2019-5642',
                   'SPT-CLJ2020-6314': 'SPT-CLJ2020-6314',
                   'SPT-CLJ2022-6323': 'SPT-CLJ2022-6324',
                   'SPT-CLJ2026-4513': 'SPT-CLJ2026-4513',
                   'SPT-CLJ2030-5638': 'SPT-CLJ2031-5638',
                   'SPT-CLJ2034-5936': 'SPT-CLJ2034-5936',
                   'SPT-CLJ2035-5251': 'SPT-CLJ2035-5251',
                   'SPT-CLJ2040-4451': 'SPT-CLJ2040-4451',
                   'SPT-CLJ2040-5342': 'SPT-CLJ2040-5342',
                   'SPT-CLJ2040-5725': 'SPT-CLJ2040-5726',
                   'SPT-CLJ2043-5035': 'SPT-CLJ2043-5035',
                   'SPT-CLJ2050-4213': 'SPT-CLJ2050-4213',
                   'SPT-CLJ2056-4405': 'SPT-CLJ2056-4405',
                   'SPT-CLJ2056-5459': 'SPT-CLJ2057-5500',
                   'SPT-CLJ2058-5608': 'SPT-CLJ2058-5609',
                   'SPT-CLJ2100-4548': 'SPT-CLJ2100-4548',
                   'SPT-CLJ2101-5542': 'SPT-CLJ2101-5542',
                   'SPT-CLJ2106-4421': 'SPT-CLJ2106-4421',
                   'SPT-CLJ2106-5844': 'SPT-CLJ2106-5845',
                   'SPT-CLJ2106-6019': 'SPT-CLJ2106-6019',
                   'SPT-CLJ2108-4445': 'SPT-CLJ2108-4445',
                   'SPT-CLJ2109-4626': 'SPT-CLJ2109-4626',
                   'SPT-CLJ2110-5244': 'SPT-CLJ2110-5244',
                   'SPT-CLJ2118-5055': 'SPT-CLJ2118-5055',
                   'SPT-CLJ2120-4728': 'SPT-CLJ2120-4728',
                   'SPT-CLJ2124-6124': 'SPT-CLJ2125-6125',
                   'SPT-CLJ2130-6458': 'SPT-CLJ2131-6459',
                   'SPT-CLJ2135-5726': 'SPT-CLJ2136-5726',
                   'SPT-CLJ2136-6307': 'SPT-CLJ2137-6307',
                   'SPT-CLJ2138-6008': 'SPT-CLJ2138-6008',
                   'SPT-CLJ2137-6437': 'SPT-CLJ2138-6437',
                   'SPT-CLJ2140-5727': 'SPT-CLJ2141-5727',
                   'SPT-CLJ2146-4633': 'SPT-CLJ2146-4632',
                   'SPT-CLJ2146-4846': 'SPT-CLJ2146-4846',
                   'SPT-CLJ2145-5644': 'SPT-CLJ2146-5645',
                   'SPT-CLJ2146-5736': 'SPT-CLJ2147-5737',
                   'SPT-CLJ2148-4843': 'SPT-CLJ2148-4843',
                   'SPT-CLJ2149-5330': 'SPT-CLJ2149-5330',
                   'SPT-CLJ2159-6244': 'SPT-CLJ2200-6245',
                   'SPT-CLJ2203-5047': 'SPT-CLJ2203-5047',
                   'SPT-CLJ2206-5807': 'SPT-CLJ2206-5808',
                   'SPT-CLJ2214-4642': 'SPT-CLJ2214-4642',
                   'SPT-CLJ2218-4519': 'SPT-CLJ2218-4519',
                   'SPT-CLJ2218-5532': 'SPT-CLJ2218-5532',
                   'SPT-CLJ2220-4534': 'SPT-CLJ2220-4534',
                   'SPT-CLJ2222-4834': 'SPT-CLJ2222-4834',
                   'SPT-CLJ2228-5828': 'SPT-CLJ2228-5828',
                   'SPT-CLJ2229-4320': 'SPT-CLJ2229-4320',
                   'SPT-CLJ2235-4416': 'SPT-CLJ2235-4416',
                   'SPT-CLJ2236-4555': 'SPT-CLJ2236-4555',
                   'SPT-CLJ2241-4558': 'SPT-CLJ2241-4558',
                   'SPT-CLJ2245-6206': 'SPT-CLJ2245-6207',
                   'SPT-CLJ2250-4808': 'SPT-CLJ2250-4808',
                   'SPT-CLJ2251-4848': 'SPT-CLJ2251-4848',
                   'SPT-CLJ2254-4907': 'SPT-CLJ2254-4907',
                   'SPT-CLJ2258-4044': 'SPT-CLJ2258-4044',
                   'SPT-CLJ2259-6057': 'SPT-CLJ2259-6057',
                   'SPT-CLJ2259-5431': 'SPT-CLJ2300-5432',
                   'SPT-CLJ2259-5617': 'SPT-CLJ2300-5617',
                   'SPT-CLJ2301-4023': 'SPT-CLJ2301-4023',
                   'SPT-CLJ2300-5331': 'SPT-CLJ2301-5331',
                   'SPT-CLJ2301-5546': 'SPT-CLJ2302-5546',
                   'SPT-CLJ2306-6505': 'SPT-CLJ2306-6505',
                   'SPT-CLJ2311-4203': 'SPT-CLJ2311-4203',
                   'SPT-CLJ2312-4621': 'SPT-CLJ2312-4621',
                   'SPT-CLJ2311-5820': 'SPT-CLJ2312-5820',
                   'SPT-CLJ2319-4716': 'SPT-CLJ2319-4716',
                   'SPT-CLJ2329-5831': 'SPT-CLJ2330-5832',
                   'SPT-CLJ2335-4243': 'SPT-CLJ2335-4244',
                   'SPT-CLJ2337-5942': 'SPT-CLJ2337-5942',
                   'SPT-CLJ2337-5912': 'SPT-CLJ2338-5912',
                   'SPT-CLJ2341-5119': 'SPT-CLJ2341-5120',
                   'SPT-CLJ2342-4714': 'SPT-CLJ2342-4714',
                   'SPT-CLJ2342-5411': 'SPT-CLJ2343-5411',
                   'SPT-CLJ2344-4243': 'SPT-CLJ2344-4243',
                   'SPT-CLJ2345-6405': 'SPT-CLJ2345-6406',
                   'SPT-CLJ2352-4657': 'SPT-CLJ2352-4657',
                   'SPT-CLJ2351-5452': 'SPT-CLJ2352-5453',
                   'SPT-CLJ2352-6134': 'SPT-CLJ2352-6134',
                   'SPT-CLJ2356-4220': 'SPT-CLJ2356-4221',
                   'SPT-CLJ2355-5055': 'SPT-CLJ2356-5056',
                   'SPT-CLJ2358-4354': 'SPT-CLJ2358-4354',
                   'SPT-CLJ2358-5229': 'SPT-CLJ2358-5229'}
common_keys = set(sptsz_id_lookup).intersection(sptpol_results)
for common_id in common_keys:
    targeted_curve = sptsz_results[sptsz_id_lookup[common_id]]
    ssdf_curve = sptpol_results[common_id]

    fig, ax = plt.subplots()
    ax.plot(mag_bins, targeted_curve, label='Targeted Obs')
    ax.plot(mag_bins, ssdf_curve, label='SSDF Cutout')
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set(title=rf'Completeness Simulations of {common_id} for IRAC 4.5 $\mu$m',
           xlabel='Vega Magnitude', ylabel='Recovery Rate')
    fig.savefig(f'Data/Comp_Sim/SPTpol/Plots/common_clusters/{common_id}_comparison.pdf', format='pdf')
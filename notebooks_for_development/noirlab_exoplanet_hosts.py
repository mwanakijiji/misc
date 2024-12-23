import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table, join, Column
import requests
import xml.etree.ElementTree as ET
import re
import ipdb
from astropy.table import MaskedColumn
from matplotlib.legend_handler import HandlerPathCollection


import astroquery
from astroquery.simbad import Simbad


def main(transit_only = False,
         rv_only = False,
         v_mag_lim = 8.5,
         dec_lim = np.array([10,60]),
         fontsize = 20):
    
    # transit_only: do you want to show only transiting systems?
    # v_mag_lim: dimmest star observable
    # dec_lim: declination limits
    # fontsize: for the plot titles, legends, axes, ticks

    stem = '/Users/bandari/Documents/git.repos/misc/notebooks_for_development/'

    neid_targets = Table.read('neid_planets.csv', format='ascii', delimiter='|')

    # Customize the Simbad query to include additional parameters
    custom_simbad = Simbad()
    custom_simbad.TIMEOUT = 120  # Increase timeout for large queries
    #custom_simbad.add_votable_fields('parallax', 'pmra', 'pmdec', 'sp')  # Add desired fields including spectral class
    custom_simbad.add_votable_fields('parallax', 'pmra', 'pmdec', 'sp', 'ids', 'flux(V)', 'flux(R)')  # Add desired fields including spectral class, identifiers, and V magnitude

    # Query Simbad using the object names from your table
    object_names = neid_targets['Name'] 
    neid_target_addl_info = custom_simbad.query_objects(object_names)
    ipdb.set_trace()

    # I want the HD designation alone, to enable table merging. 
    # This function is to extract the string containing 'HD ' from the 'ids' field
    def extract_hd_id(ids):
        for id_str in ids.split('|'):
            if 'HD ' in id_str:
                return id_str
        return None

    # make a new col just containing the HD name
    neid_target_addl_info['HD_NAME'] = [extract_hd_id(ids) for ids in neid_target_addl_info['IDS']]
    # normalize for merging
    neid_target_addl_info['HD_NORM'] = [hd.replace(' ', '') for hd in neid_target_addl_info['HD_NAME']]
    #neid_target_addl_info['NORM_ID'] = [hd.replace(' ', '') for hd in neid_target_addl_info['SIMBAD_HD']]

    # make col of same name to allow merge
    neid_targets['HD_NORM'] = neid_targets['Name']

    # merged table
    neid_target_list_w_addl_info = join(neid_targets, neid_target_addl_info, keys='HD_NORM')

    # get stellar radii
    # define spectral class properties with *approx* radii
    # (normalized to solar radius)
    spectral_to_radius = {
        'O': 6.6,
        'B': 4.2,
        'A': 1.6,
        'F': 1.3,
        'G': 1.0,
        'K': 0.85,
        'M': 0.5
    }

    # Add a radius column based on the spectral class
    def get_radius(spectral_class):
        # Extract the main spectral type (e.g., 'A' from 'A3V')
        main_type = spectral_class[0].upper()
        return spectral_to_radius.get(main_type, np.nan)  # Return NaN if not in the mapping

    # Apply the function to the table
    neid_target_list_w_addl_info['Radius'] = [get_radius(sc) for sc in neid_target_list_w_addl_info['SP_TYPE']]  # Replace 'Spectral_Class'

    # distance in pc
    neid_target_list_w_addl_info['dist'] = 1/(1e-3 * neid_target_list_w_addl_info['PLX_VALUE'])

    plt.clf()
    plt.hist(neid_target_list_w_addl_info['dist'], bins=20)
    plt.title('NEID target list distances', fontsize=fontsize)
    plt.xlabel('Dist (pc)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()

    plt.clf()
    plt.scatter(neid_target_list_w_addl_info['dist'], neid_target_list_w_addl_info['Radius'])
    plt.title('NEID target list stellar radii', fontsize=fontsize)
    plt.xlabel('Dist (pc)', fontsize=fontsize)
    plt.ylabel('Radius (R_sol)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()

    # find angular widths
    neid_target_list_w_addl_info['width_ang'] = ( 2 * neid_target_list_w_addl_info['Radius'] / neid_target_list_w_addl_info['dist'] ) * (1./107.) # arcsec (N.b. 1 AU = 107 R_sol)

    plt.clf()
    plt.hist(1000 * neid_target_list_w_addl_info['width_ang'], bins=20)
    plt.title('NEID target list angular sizes (approx)', fontsize=fontsize)
    plt.xlabel('Stellar diameter (masec)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()

    # overlap of NEID targets with JMDC catalog (i.e., directly measured angular diameters)
    jmdc_catalog = Table.read('jmdc_catalog.csv', format='ascii', delimiter='|', data_start=44)
    # to enable merger
    jmdc_catalog['HD_NORM'] = jmdc_catalog['ID1']
    # inner merger to get table of NEID targets observed with OLBI
    neid_targets_observed = join(neid_target_list_w_addl_info, jmdc_catalog, keys='HD_NORM', join_type='inner')
    neid_targets_observed = neid_targets_observed.group_by('Name').groups.aggregate(np.mean)
    ipdb.set_trace()



    # make DataFrame to make masks, annotations easier
    df_neid_targets_observed = neid_targets_observed.to_pandas()
    df_neid_target_list_w_addl_info = neid_target_list_w_addl_info.to_pandas()
    ipdb.set_trace()

    mask_all_neid_systems_bright = (df_neid_target_list_w_addl_info['Vmag'] <= v_mag_lim)
    mask_all_neid_systems_too_dim = (df_neid_target_list_w_addl_info['Vmag'] > v_mag_lim)
    ipdb.set_trace()

    marker_scale_factor_neid = 1e4
    linear_marker_sizes_neid_bright = marker_scale_factor_neid * np.power(10,(-df_neid_target_list_w_addl_info['Vmag'][mask_all_neid_systems_bright].values/2.5))
    linear_marker_sizes_jmdc = marker_scale_factor_neid * np.power(10,(-neid_targets_observed['Vmag']/2.5))
    ipdb.set_trace()

    plt.clf()
    plt.title('NEID targets', fontsize=fontsize)
    # not interferometrically observed (actually, points will get overplotted if already interferometrically observed)
    plt.scatter(neid_target_list_w_addl_info['dist'][mask_all_neid_systems_bright], 1000 * neid_target_list_w_addl_info['width_ang'][mask_all_neid_systems_bright], 
                color='blue', 
                s = linear_marker_sizes_neid_bright,
                edgecolor='black')
    # interferometrically observed
    plt.scatter(neid_targets_observed['dist'], 1000 * neid_targets_observed['width_ang'], 
                color='red', 
                s = linear_marker_sizes_jmdc,
                edgecolor='black')
    # too dim
    plt.scatter(neid_target_list_w_addl_info['dist'][mask_all_neid_systems_too_dim], 1000 * neid_target_list_w_addl_info['width_ang'][mask_all_neid_systems_too_dim], 
                color='gray', 
                s = 40,
                edgecolor='black', 
                label='Too dim')
    # for the legend
    plt.scatter(-999, -999, 
                s=40, 
                color='blue', edgecolor='black', label='Not interferometrically observed', zorder=3)
    # for the legend
    plt.scatter(-999, -999, 
                s=40, 
                color='red', edgecolor='black', label='Observed', zorder=3)
    plt.fill_betweenx([0, 0.530], x1=0, x2=45, color='gray', alpha=0.3) # longest wavel of CHARA/VEGA is 530 nm
    plt.xlim([0,45])
    plt.ylim([0,5.5])
    plt.xlabel('Dist (pc)', fontsize=fontsize)
    plt.ylabel('Rough angular width (mas)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    leg = plt.legend(fontsize=fontsize, loc='upper right', fancybox=True, framealpha=1)
    leg.set_zorder(10)
    plt.show()
    ipdb.set_trace()

    plt.clf()
    plt.title('NEID targets', fontsize=fontsize)
    plt.scatter(neid_target_list_w_addl_info['dist'], neid_target_list_w_addl_info['Radius'], color='blue', label='Not interferometrically observed')
    plt.scatter(neid_targets_observed['dist'], neid_targets_observed['Radius'], color='red', label='Observed')
    plt.xlabel('Dist (pc)', fontsize=fontsize)
    plt.ylabel('Radius (R_sol)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    leg = plt.legend(fontsize=fontsize, loc='upper right', fancybox=True, framealpha=1)
    leg.set_zorder(10)
    plt.show()

    # for comparison: predicted diameter of HD86728
    # Boyajian+ 2012 measured 0.753 mas
    #float(neid_target_list_w_addl_info['width_ang'][neid_target_list_w_addl_info['HD_NORM'] == 'HD86728'])

    # overlap of transiting planets with JMDC catalog (i.e., directly measured angular diameters)
    # read in catalog
    all_planets_catalog = Table.read('all_planets_20241220.csv', format='ascii', delimiter=',')
    # Check if 'Name' is a MaskedColumn (in which case we cannot directly replace strings and normalize the HD designations)
    if isinstance(all_planets_catalog['hd_name'], MaskedColumn):
        # Replace masked values with a default string, e.g., 'Unknown'
        #print('yes')
        filled_name = all_planets_catalog['hd_name'].filled('Unknown')
        # Create a new Column from the filled data
        #print(Column(filled_name))
        all_planets_catalog['hd_name'] = Column(filled_name)
        #print(all_planets_catalog['hd_name'])
    else:
        # If it's already a Column, no conversion is needed
        all_planets_catalog['hd_name'] = all_planets_catalog['hd_name']

    # to enable merger
    all_planets_catalog['HD_NORM'] = [hd.replace(' ', '') for hd in all_planets_catalog['hd_name']]

    # use pandas to make the following easier
    df_all_planets_catalog = all_planets_catalog.to_pandas()

    # remove all rows not relevant to transits
    if transit_only == True:
        df_all_planets_catalog = df_all_planets_catalog[df_all_planets_catalog['tran_flag'] == 1]
        plot_title = 'Transiting exoplanet systems'
        marker_scale_factor = 1e5 # for plotting
        print('Transiting systems only')
    elif rv_only == True:
        df_all_planets_catalog = df_all_planets_catalog[df_all_planets_catalog['rv_flag'] == 1]
        plot_title = 'RV exoplanet systems'
        marker_scale_factor = 1e5 # for plotting
        print('RV systems only')
    elif transit_only == False & rv_only == False:
        plot_title = 'All nearby exoplanet host stars'
        print('All exoplanet systems')

    # Group by the 'HD_NORM' column and aggregate using nanmedian for numerical columns, and first value in string columns
    df_all_systems_catalog = df_all_planets_catalog.groupby('HD_NORM').agg(lambda x: np.nanmedian(x) if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    # Reset the index to turn the grouped column back into a regular column
    df_all_systems_catalog.reset_index(inplace=True)

    # remove rows with controversial flag
    df_all_systems_catalog = df_all_systems_catalog[df_all_systems_catalog['pl_controv_flag'] != 1]
    df_all_systems_catalog.reset_index(inplace=True)

    # find approx. angular widths of stars
    df_all_systems_catalog['width_ang'] = ( 2 * df_all_systems_catalog['st_rad'] / df_all_systems_catalog['sy_dist'] ) * (1./107.) # arcsec (N.b. 1 AU = 107 R_sol)

    # convert to pandas to enable merger
    df_jmdc_catalog = jmdc_catalog.to_pandas()

    # merge
    df_all_systems_with_jmdc_measurements = pd.merge(df_all_systems_catalog, df_jmdc_catalog, on='HD_NORM', how='inner')

    # make cut based on DEC
    df_all_systems_catalog_cut = df_all_systems_catalog[np.logical_and(df_all_systems_catalog['dec'] <= dec_lim[1], df_all_systems_catalog['dec'] > dec_lim[0])]
    df_all_systems_catalog_cut.reset_index(inplace=True)
    print('DEC cut:',dec_lim)

    # make cut to larger catalog based on brightness (V<8.5)
    '''
    import ipdb; ipdb.set_trace()
    df_all_systems_catalog_cut = df_all_systems_catalog_cut[df_all_systems_catalog_cut['sy_vmag'] <= v_mag_lim]
    df_all_systems_catalog_cut.reset_index(inplace=True)
    df_all_systems_catalog_too_dim = df_all_systems_catalog[df_all_systems_catalog['sy_vmag'] > v_mag_lim] # these are probably too dim
    df_all_systems_catalog_too_dim.reset_index(inplace=True)
    '''
    print('Vmag cut:',v_mag_lim)

    # reassign
    df_all_systems_catalog = df_all_systems_catalog_cut.copy(deep=True)

    # plot of all exoplanet host stars, and those with interferometric measurements
    plt.clf()
    plt.title(plot_title, fontsize = 20) # with HD designation and known (at least approx.) stellar angular widths

    # Filter for finite angular width values and sufficient brightness
    finite_mask_all_systems_bright = (
        np.isfinite(df_all_systems_catalog['sy_dist']) & 
        np.isfinite(df_all_systems_catalog['width_ang']) & 
        (df_all_systems_catalog['sy_vmag'] <= v_mag_lim)
    )
    finite_mask_all_systems_too_dim = (
        np.isfinite(df_all_systems_catalog['sy_dist']) & 
        np.isfinite(df_all_systems_catalog['width_ang']) & 
        (df_all_systems_catalog['sy_vmag'] > v_mag_lim)
    )

    finite_mask_jmdc = np.isfinite(df_all_systems_with_jmdc_measurements['sy_dist']) & np.isfinite(df_all_systems_with_jmdc_measurements['width_ang'])

    linear_marker_sizes_all_bright = marker_scale_factor * np.power(10,(-df_all_systems_catalog['sy_vmag'][finite_mask_all_systems_bright].values/2.5))
    linear_marker_sizes_jmdc = marker_scale_factor * np.power(10,(-df_all_systems_with_jmdc_measurements['sy_vmag'][finite_mask_jmdc]/2.5))
    # plot sufficiently bright systems, with markers sized according to brightness 
    # not interferometrically observed (actually, points will get overplotted if already interferometrically observed)
    plt.scatter(df_all_systems_catalog['sy_dist'][finite_mask_all_systems_bright], 1000 * df_all_systems_catalog['width_ang'][finite_mask_all_systems_bright], 
                s=linear_marker_sizes_all_bright, 
                color='blue', edgecolor='black', zorder=3)
    # for the legend
    plt.scatter(-999, -999, 
                s=40, 
                color='blue', edgecolor='black', label='Not interferometrically observed', zorder=3)
    # interferometrically observed
    plt.scatter(df_all_systems_with_jmdc_measurements['sy_dist'][finite_mask_jmdc], 1000 * df_all_systems_with_jmdc_measurements['width_ang'][finite_mask_jmdc],
                s=linear_marker_sizes_jmdc, 
                color='red', edgecolor='black', zorder=4)
    # for the legend
    plt.scatter(-999, -999, 
                s=40, 
                color='red', edgecolor='black', label='Observed', zorder=4)
    # too dim
    plt.scatter(df_all_systems_catalog['sy_dist'][finite_mask_all_systems_too_dim], 1000 * df_all_systems_catalog['width_ang'][finite_mask_all_systems_too_dim], 
                s=40, 
                color='gray')
    # for the legend
    plt.scatter(-999, -999, 
                s=40, 
                color='gray', label='Too dim', zorder=4)
    # annotate with names of host stars bright enough for interferometry
    for i in range(len(df_all_systems_catalog[finite_mask_all_systems_bright])):
        plt.annotate(df_all_systems_catalog['hostname'][finite_mask_all_systems_bright].iloc[i],
                (df_all_systems_catalog['sy_dist'][finite_mask_all_systems_bright].iloc[i], 
                1000 * df_all_systems_catalog['width_ang'][finite_mask_all_systems_bright].iloc[i]),
                xytext = (df_all_systems_catalog['sy_dist'][finite_mask_all_systems_bright].iloc[i] + 1, 
                          1000 * df_all_systems_catalog['width_ang'][finite_mask_all_systems_bright].iloc[i] + 0.3),
                fontsize=8, ha='left', va='bottom',
                arrowprops=dict(arrowstyle='-', color='gray'), zorder=3)
    plt.fill_betweenx([0, 0.530], x1=0, x2=np.max(df_all_systems_catalog['sy_dist']), color='gray', alpha=0.3, zorder=1) # longest wavel of CHARA/VEGA is 530 nm
    plt.fill_betweenx([0, 0.150], x1=0, x2=np.max(df_all_systems_catalog['sy_dist']), color='gray', alpha=0.3, zorder=2)
    plt.xlim([0,200])
    plt.ylim([0,5])
    plt.xlabel('Dist (pc)', fontsize=fontsize)
    plt.ylabel('Approx angular width (mas)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # fix the sizes of the markers in the legend
    #plt.legend(fontsize=fontsize, loc='upper right', fancybox=True, framealpha=1)
    #plt.show()
    # Draw the legend on top of everything else
    leg = plt.legend(fontsize=fontsize, loc='upper right', fancybox=True, framealpha=1)
    leg.set_zorder(10)
    plt.show()
    #plt.savefig('junk.png')

if __name__ == "__main__":
    main(transit_only = False, 
         rv_only = True,
         v_mag_lim = 8.5,
         dec_lim = np.array([10,60]),
         fontsize = 20)

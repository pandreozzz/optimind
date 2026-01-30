import numpy as np
import xarray as xr

from ..main.config import CONFIGDICT, SOME_AEROS_OUT_OF_CLOUD

def compute_lut_species_from_ifs_species(recipes, aero_cams,
                                         densities, scale_mmr=None):
    """
        Recipes can be dict or list
    """
    lut_species = {}
    if isinstance(recipes, dict):
        recipes_dict = recipes
    elif isinstance(recipes, list):
        recipes_dict = {f"aero{a+1}": rec
                        for a,rec in enumerate(recipes)}
    else:
        raise ValueError("Error in compute_lut_species_from_ifs_species: "+\
                         f"unexpected type for recipes ({type(recipes)})")
        
    for r,(ccn_name,recipe) in enumerate(recipes_dict.items()):
        n_valid_ingredients = 0
        this_ingredients = []
        for aero in recipe:
            if aero in aero_cams:
                this_ingredients.append(
                    (recipe[aero]*aero_cams[aero]).expand_dims(dim="tmp_ingredient"))
                n_valid_ingredients += 1
            else:
                print(f"Warning while computing lut species {ccn_name}: "+\
                      f"species not represented in fields: {aero}")

        if n_valid_ingredients > 0:
            this_bowl = xr.concat(this_ingredients, coords="all",
                                  dim="tmp_ingredient").sum(dim="tmp_ingredient")
        else:
            this_bowl = xr.DataArray(0.)

        scale_factor = 1 if scale_mmr is None else scale_mmr[r]
        lut_species[ccn_name] = (this_bowl*densities*scale_factor).rename(f"{ccn_name}_mcon")

        del this_bowl

    return xr.Dataset(lut_species)

def compute_ccn_species(this_aero : xr.Dataset, this_recipe) -> xr.Dataset:

    this_lut_species =\
    compute_lut_species_from_ifs_species(
        recipes=this_recipe,
        aero_cams=this_aero,
        densities=np.float32(1.), scale_mmr=None)

    return this_lut_species.clip(min=0.)


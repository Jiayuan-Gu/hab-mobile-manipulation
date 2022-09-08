# https://github.com/facebookresearch/habitat-sim/blob/master/conda-build/habitat-sim/build.sh

# site packages dir
if [ -z "$PREFIX" ]; then
  PREFIX=./venv/lib/python3.6/site-packages
  # PREFIX=~/miniconda3/lib/python3.6/site-packages
fi

pushd "${PREFIX}" || exit
corrade_bindings=$(find . -name "*_corrade*so")
echo "${corrade_bindings}"
magnum_bindings=$(find . -name "*_magnum*so")
echo "${magnum_bindings}"
hsim_bindings=$(find . -name "*habitat_sim_bindings*so")
echo "${hsim_bindings}"
ext_folder=$(dirname "${hsim_bindings}")
echo "${ext_folder}"

# Adding rpath for everything to have both habitat_sim/_ext and the conda env's lib dir
# All this is done relatively to the *.so's folder to make it relocatable
patchelf --set-rpath "\$ORIGIN/${ext_folder}" --force-rpath "${corrade_bindings}"
patchelf --set-rpath "\$ORIGIN/${ext_folder}" --force-rpath "${magnum_bindings}"
patchelf --set-rpath "\$ORIGIN" --force-rpath "${hsim_bindings}"

find "${ext_folder}" -name "*Corrade*so" -print0 | xargs --null -I {} patchelf --set-rpath "\$ORIGIN" --force-rpath {}

pushd "$(dirname "${corrade_bindings}")/corrade" || exit
find . -name "*so" -print0 | xargs --null -I {} patchelf --set-rpath "\$ORIGIN/../${ext_folder}" --force-rpath {}
popd || exit

popd || exit

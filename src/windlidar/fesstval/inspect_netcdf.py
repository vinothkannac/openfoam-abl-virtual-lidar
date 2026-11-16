"""List dimensions, variables, units and long names of a netCDF file, to find the variable names of a data product.
Usage: python -m windlidar.fesstval.inspect_netcdf file.nc
"""
import sys
import xarray as xr
def main(path):
    ds = xr.open_dataset(path, decode_times=True)
    print(ds)
    for n, v in ds.variables.items():
        print("%-28s dims=%-28s units=%-10s long_name=%s" % (n, v.dims, v.attrs.get("units", ""), v.attrs.get("long_name", "")))


if __name__ == "__main__":
    main(sys.argv[1])

import os


file_path = os.path.dirname(os.path.realpath(__file__))


TEMPLATE = """[time_unit]
frequency = 1

[topology]
chiplet_number = 4
chiplet_rows = 2
chiplet_columns = 2
core_number = 16
core_rows = 4
core_columns = 4
tensorcore_number = 1
vectorunit_number = 1
sram_number = 1
ddr_number = 1
topology_type = tlm

[tensorcore_cfg]
tensorcore_type = multree
tensorcore_grain = [128, 64]
tensorcore_timeunit = 1

[vectorunit_cfg]
vectorunit_type = simd
vectorunit_grain = [1024]
vectorunit_timeunit = 1
eleadd_complexity = 1
eleexp_complexity = 2
elegelu_complexity = 8
elemul_complexity = 1
elepow2_complexity = 1
elerelu_complexity = 2
elesqrt_complexity = 4
matadd_complexity = 1
redmax_complexity = 1
redsum_complexity = 1
vecadd_complexity = 1
vecdiv_complexity = 4
vecmac_complexity = 1
vecmul_complexity = 1
transpose_complexity = 1
lookup_complexity = 1

[co2co_cfg]
co2co_latency = 1
co2co_bandwidth = 256
co2co_timeunit = 1

[ch2ch_cfg]
ch2ch_latency = 20
ch2ch_bandwidth = 256
ch2ch_timeunit = 1

[sram_cfg]
sram_capacity = 16777216

[ddr_cfg]
ddr_capacity = 34359738368

[co2ddr_cfg]
co2ddr_latency = 50
co2ddr_bandwidth = 256
co2ddr_timeunit = 1

[ddr2co_cfg]
ddr2co_latency = 50
ddr2co_bandwidth = 256
ddr2co_timeunit = 1

[failures]
failed_nodes = []
failed_links = []
"""


def main():
    output_dir = os.path.join(file_path, "cfg")
    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(
        output_dir,
        "config_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.cfg",
    )
    with open(cfg_path, "w", encoding="utf-8") as handle:
        handle.write(TEMPLATE)
    print(f"Written {cfg_path}")


if __name__ == "__main__":
    main()


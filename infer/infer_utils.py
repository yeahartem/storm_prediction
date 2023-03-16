# print("Sample maps")
#     for i, time in enumerate(gdf.time.unique()):
#         f, ax = plt.subplots(1, figsize=(10, 5))
#         ax = gdf[gdf['time'] == time].plot(column='prob', cmap='afmhot', ax=ax, legend=True)
#         # if i > 10:
#         #     break
#         plt.savefig(os.path.join(path_to_save, 'pics', str(time) + '.png'))
#     print("Sample maps (10) - done")
#     logging.info("Sample maps (10)")
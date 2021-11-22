##
using Images
using ImageView
using ImageBinarization
using Makie
using GLMakie
# using Plots
# plotlyjs()

include("mappingFunctions.jl")

## Data
rgb1 = load("sampleImages/rgb1.png");
rgb2 = load("sampleImages/rgb2.png");
d8 = load("sampleImages/depth8.png");
d16 = load("sampleImages/depth16.png");

""" Given Box Annotation 
"regions": [
    "boundingBox": {
            "height": 262.94117647058823,
            "width": 395.20330806340456,
            "left": 471.9503790489318,
            "top": 248.90625
    },
    "points": [
        {
            "x": 471.9503790489318,
            "y": 248.90625
        },
        {
            "x": 546.0509993108201,
            "y": 301.84742647058823
        },
        {
            "x": 648.3804272915231,
            "y": 363.61213235294116
        },
        {
            "x": 758.6492074431427,
            "y": 420.96507352941177
        },
        {
            "x": 818.6354238456237,
            "y": 458.90625
        },
        {
            "x": 867.1536871123363,
            "y": 511.84742647058823
        }
        ]
    }
"""

gt_box_xywh = Int32.(round.(Float32[471.9503, 248.9062, 395.2033, 262.9411]))

gt_polyline = [
    Float32[471.9503, 248.9062],
    Float32[546.0509, 301.8474],
    Float32[648.3804, 363.6121],
    Float32[758.6492, 420.9650],
    Float32[818.6354, 458.9062],
    Float32[867.1536, 511.8474]
]

## calculate panel mask on rgb1
# panel_mask = binarize(Gray.(rgb1),Otsu())
# Balance Yield higher panel uniform over Otsu
panel_mask = binarize(Gray.(rgb1), Balanced());

d8_processed = channelview(d8)[1, :, :];
d16_processed = channelview(d16) .* 10; # x10 for visualization 

# depth image selection
depthmap = d16_processed;

# panel-mask pixels
panel_pixels = findall(Bool.(panel_mask));

# panel masked by bbbox
bbox_include = false
if bbox_include
    panel_pixels = filter(panel_pixels) do x
        x[1] in gt_box_xywh[2]:(gt_box_xywh[2]+gt_box_xywh[4]) &&
            x[2] in gt_box_xywh[1]:(gt_box_xywh[1]+gt_box_xywh[3])
    end
end

# depth image look-up
depth_values = depthmap[panel_pixels];

# 3D Points
pts = rs2deproject_px_to_pt.(Ref(d435intrx), panel_pixels, depth_values) |> unique;
# filter out zero depth
pts = filter(x -> x[3] != 0, pts);

# split vector to column-array for plots
function split_pts(pts::Vector{Vector{Float32}})
    pts_x = Float32[]
    pts_y = Float32[]
    pts_z = Float32[]
    for i in pts
        push!(pts_x, i[1])
        push!(pts_y, i[2])
        push!(pts_z, i[3])
    end
    return pts_x, pts_y, pts_z
end

# scene pointcloud
pts_x, pts_y, pts_z = split_pts(pts);
# scatter(pts_x, pts_y, pts_z, markersize = 1, show_axis = true)

##
# Plot Polyline Annotation on 3D space
gt_polyline_px = map(x -> Int.(round.(x)), gt_polyline)
gt_polyline_px = map(gt_polyline_px) do x
    return CartesianIndex(x[2], x[1])
end
gt_polyline_depth = depthmap[gt_polyline_px]
gt_polyline_3d = rs2deproject_px_to_pt.(Ref(d435intrx), gt_polyline_px, gt_polyline_depth)

# 3d Polyline 
gt_pts_x, gt_pts_y, gt_pts_z = split_pts(gt_polyline_3d);
# scatterlines!(gt_pts_x, gt_pts_y, gt_pts_z, markersize = 2, markercolor = :red, color = :red)

##
# Linear Interpolation in polyline intervals
using BasicInterpolators

# interpolated polyline
pts_itp_x = Int32[];
pts_itp_y = Int32[];

# for each interval
interval_count = length(gt_polyline_px) - 1

for i = 1:interval_count
    data_y = [gt_polyline_px[i].I[1], gt_polyline_px[i+1].I[1]] # row
    data_x = [gt_polyline_px[i].I[2], gt_polyline_px[i+1].I[2]] # col

    range_x = data_x[1]:data_x[2]
    # broadcast call interpolator on x vector
    data_y_itp = LinearInterpolator(data_x, data_y).(range_x)
    data_y_itp = Int32.(round.(data_y_itp))

    # stack result ...
    push!(pts_itp_x, Int32.(collect(range_x))...)
    push!(pts_itp_y, data_y_itp...)
end

px_polyline_itp = map(pts_itp_x, pts_itp_y) do a, b
    CartesianIndex(b, a)
end
px_polyline_itp_depth = depthmap[px_polyline_itp];
polyline_itp_3d = rs2deproject_px_to_pt.(Ref(d435intrx), px_polyline_itp, px_polyline_itp_depth) |> unique
# interpolated 3d polyline
pts_itp_x, pts_itp_y, pts_itp_z = split_pts(polyline_itp_3d);
# scatterlines!(pts_itp_x, pts_itp_y, pts_itp_z, markersize = 2, markercolor = :blue, color = :blue)



""" Makie Plot"""
## Plot
fig = Figure(resolution = (1280, 720))

Label(fig[1, 1, Top()], "3D")
scatter(fig[1, 1], pts_x, pts_y, pts_z, markersize = 1.5, show_axis = true)
scatterlines!(gt_pts_x, gt_pts_y, gt_pts_z, markersize = 3, markercolor = :red, color = :red)
scatterlines!(pts_itp_x, pts_itp_y, pts_itp_z, markersize = 3, markercolor = :blue, color = :blue)

a3d = Makie.Axis3(fig[1, 2][1, 1],
    xlabel = "X",
    ylabel = "Y",
    zlabel = "Z",
    xlabelsize = 12,
    ylabelsize = 12,
    zlabelsize = 12,
    title = "XY",
    titlesize = 12,
    xticklabelsize = 10,
    yticklabelsize = 10,
    zticklabelsize = 10
)
scatterlines!(a3d, gt_pts_x, -gt_pts_y, gt_pts_z, markersize = 1.25, markercolor = :red, color = :red)
scatterlines!(a3d, pts_itp_x, -pts_itp_y, pts_itp_z, markersize = 1.25, markercolor = :blue, color = :blue)


axy = Makie.Axis(fig[1, 2][2, 1],
    xlabel = "X",
    ylabel = "Y",
    xlabelsize = 12,
    ylabelsize = 12,
    title = "XY",
    titlesize = 12,
    xticklabelsize = 10,
    yticklabelsize = 10
)
scatterlines!(axy, gt_pts_x, -gt_pts_y, markersize = 1.25, markercolor = :red, color = :red)
scatterlines!(axy, pts_itp_x, -pts_itp_y, markersize = 1.25, markercolor = :blue, color = :blue)

axz = Makie.Axis(fig[1, 2][3, 1],
    xlabel = "X",
    ylabel = "Z",
    xlabelsize = 12,
    ylabelsize = 12,
    title = "XZ",
    titlesize = 12,
    xticklabelsize = 10,
    yticklabelsize = 10,
)
scatterlines!(axz, gt_pts_x, gt_pts_z, markersize = 1.25, markercolor = :red, color = :red)
scatterlines!(axz, pts_itp_x, pts_itp_z, markersize = 1.25, markercolor = :blue, color = :blue)

azy = Makie.Axis(fig[1, 2][4, 1],
    xlabel = "Z",
    ylabel = "Y",
    xlabelsize = 12,
    ylabelsize = 12,
    title = "ZY",
    titlesize = 12,
    xticklabelsize = 10,
    yticklabelsize = 10
)
scatterlines!(azy, gt_pts_z, -gt_pts_y, markersize = 1.25, markercolor = :red, color = :red)
scatterlines!(azy, pts_itp_z, -pts_itp_y, markersize = 1.25, markercolor = :blue, color = :blue)

colsize!(fig.layout, 2, Relative(2 / 7))

fig
## 
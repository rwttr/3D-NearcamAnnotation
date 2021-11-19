##
using Images
using ImageView
using ImageBinarization
using Makie
using GLMakie
# using Plots
# plotlyjs()

include("mappingFunctions.jl")

##
rgb1 = load("sampleImages/rgb1.png");
rgb2 = load("sampleImages/rgb2.png");
d8 = load("sampleImages/depth8.png");
d16 = load("sampleImages/depth16.png");

# calculate panel mask on rgb1
# panel_mask = binarize(Gray.(rgb1),Otsu())
# Balance Yield higher panel uniform over Otsu
panel_mask = binarize(Gray.(rgb1), Balanced());

d8_processed = channelview(d8)[1, :, :];

# panel-mask pixels
panel_pixels = findall(Bool.(panel_mask));
depth_values = d8_processed[panel_pixels];

# remap depth_value to active depth range


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

pts_x, pts_y, pts_z = split_pts(pts);

scatter(pts_x, pts_y, pts_z, markersize = 1, show_axis = true)
# scatter3d(pts_x, pts_y, pts_z, size = (900, 600))

##
# Plot Polyline Annotation on 3D space
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

gt_box_xywh = Float32[471.9503, 248.9062, 395.2033, 262.9411]


gt_polyline = [
    Float32[471.9503, 248.9062],
    Float32[546.0509, 301.8474],
    Float32[648.3804, 363.6121],
    Float32[758.6492, 420.9650],
    Float32[818.6354, 458.9062],
    Float32[867.1536, 511.8474]
]


gt_polyline_px = map(x -> Int.(round.(x)), gt_polyline)
gt_polyline_px = map(gt_polyline_px) do x
    return CartesianIndex(x[2], x[1])
end
gt_polyline_depth = d8_processed[gt_polyline_px]

gt_polyline_3d = rs2deproject_px_to_pt.(Ref(d435intrx), gt_polyline_px, gt_polyline_depth)

gt_pts_x, gt_pts_y, gt_pts_z = split_pts(gt_polyline_3d);

scatterlines!(gt_pts_x, gt_pts_y, gt_pts_z, markersize = 2, markercolor = :red, color = :red)

##
# Linear Interpolation in polyline interval
using Interpolations

# for each interval
data_x = [gt_polyline_px[1].I[1], gt_polyline_px[2].I[1]]
data_y = [gt_polyline_px[1].I[2], gt_polyline_px[2].I[2]]

range_x = data_x[1]:data_x[2]
# broadcast call interpolator on x vector
data_y_itp = LinearInterpolator(data_x, data_y).(range_x)
data_y_itp = Int.(round.(data_y_itp))
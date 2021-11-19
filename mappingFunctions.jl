struct CameraIntrinsic
    source::String  # Source camera format
    ppx::Float32    # principal point x axis
    ppy::Float32    # principal point y axis
    fx::Float32     # focal length in x axis
    fy::Float32     # focal length in y axis
end

struct CameraExtrinsic
    source::String      # Source camera
    target::String      # target camera to be transform to
    trans_vec::Vector   # translation vector
    rotate_mat::Matrix  # rotational matrix
end

d435intrx = CameraIntrinsic("rgb8",
    642.823,
    352.496,
    923.531,
    923.956
);

function rs2deproject_px_to_pt(intrx::CameraIntrinsic, pxls::CartesianIndex, depthValue)
    # pxls : in (row,col) convention
    temp_x = (pxls.I[2] - intrx.ppx) / intrx.fx
    temp_y = (pxls.I[1] - intrx.ppy) / intrx.fy

    point_x = depthValue * temp_x
    point_y = depthValue * temp_y
    point_z = depthValue
    return [point_x, point_y, point_z]
end

function rs2transform_pt_to_pt(extrx::CameraExtrinsic, source_point::Vector)
    target_point = (extrx.rotate_mat * source_point) + extrx.trans_vec
    return target_point
end

# Broadcast use
# rs2deproject_px_to_pt.(Ref(d435intrx),[px1,px2],[10,20])
# VMD visualization — reference structure + SDF isosurfaces
# Run with: vmd -e visualize.tcl
#
# Change these to switch systems:
set xyzfile  "../../../ref_Pneutral_swapxz.xyz"
set dxfiles  {sdf-cl.dx sdf-na.dx sdf-np.dx sdf-pp.dx}
set isovalue 1.0

# Physical radii (sigma/2, in Angstrom) — from faunus atom definitions
# VDW style uses scale=1.0 so the stored radius IS the rendered radius
array set type_radius {MP 20.0  PP 2.0  NP 2.0  Na 2.0  Cl 2.0  UP 2.0}

# ColorID per particle type
array set type_color  {MP 6    PP 1    NP 0    Na 3    Cl 7    UP 8 }

# Material per particle type
array set type_material  {MP Opaque    PP Edgy    NP Edgy    Na Edgy    Cl Edgy   UP Edgy }

# ColorID per SDF file
array set dx_color {
    sdf-cl.dx 7
    sdf-na.dx 3
    sdf-np.dx 0
    sdf-pp.dx 1
}

# --- Parse xyz to find which types are present ---
set fp [open $xyzfile r]
gets $fp natoms
gets $fp _comment
set found_types {}
for {set i 0} {$i < $natoms} {incr i} {
    gets $fp line
    set t [lindex $line 0]
    if {[lsearch -exact $found_types $t] < 0} {
        lappend found_types $t
    }
}
close $fp

# --- Display settings ---
axes location Off
color Display Background white
light 2 on
light 3 on
display projection Orthographic

# --- Load structure + volumetric data ---
mol new $xyzfile type xyz
set molid [molinfo top get id]
foreach dx $dxfiles {
    mol addfile $dx type dx mol $molid
}

# Set physical radii — VDW scale=1.0, so radius = actual rendered size
foreach t $found_types {
    if {[info exists type_radius($t)]} {
        set sel [atomselect top "type $t"]
        $sel set radius $type_radius($t)
        $sel delete
    }
}

# --- Structural representations (VDW scale=1.0, radius drives the size) ---
mol delrep 0 top
set rep 0
foreach t $found_types {
    if {![info exists type_color($t)]} continue
    mol addrep top
    mol modselect   $rep top "type $t"
    mol modstyle    $rep top VDW 1.0 32.0
    mol modcolor    $rep top ColorID $type_color($t)
    mol modmaterial $rep top $type_material($t)
    incr rep
}

# --- SDF isosurfaces ---
set vol_idx 0
foreach dx $dxfiles {
    set cid [expr {[info exists dx_color($dx)] ? $dx_color($dx) : 7}]
    mol addrep top
    mol modstyle    $rep top Isosurface $isovalue $vol_idx 0 0 1 1
    mol modcolor    $rep top ColorID $cid
    mol modmaterial $rep top Transparent
    incr rep
    incr vol_idx
}

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import scipy

st.set_page_config(page_title="Beamforming & MIMO Simulator", layout="wide", page_icon="📡")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📡 Beamforming & MIMO Simulator</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("MIMO & Beamforming")
app_mode = st.sidebar.selectbox(
    "Choose Module",
    [
        "📶 Antenna Array Patterns",
        "🎯 Digital Beamforming",
        "📡 Beam Steering",
        "🔢 MIMO Capacity",
        "🌐 Massive MIMO",
        "📊 Beam Management"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Explore antenna arrays, beamforming, beam steering, and MIMO systems. "
    "Understand how 5G uses multiple antennas for massive capacity gains."
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def uniform_linear_array_pattern(N, d_lambda, theta_deg, theta_steer_deg=0):
    """
    Calculate array factor for uniform linear array
    N: number of elements
    d_lambda: element spacing in wavelengths
    theta_deg: scan angles in degrees
    theta_steer_deg: steering angle in degrees
    """
    theta = np.deg2rad(theta_deg)
    theta_steer = np.deg2rad(theta_steer_deg)
    
    # Array factor
    psi = 2 * np.pi * d_lambda * (np.cos(theta) - np.cos(theta_steer))
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        AF = np.abs(np.sin(N * psi / 2) / (N * np.sin(psi / 2)))
        AF = np.where(np.isnan(AF), N, AF)  # Set to N at theta = theta_steer
    
    # Normalize
    AF = AF / N
    
    return AF

def uniform_planar_array_pattern(Nx, Ny, dx_lambda, dy_lambda, theta_deg, phi_deg, theta_steer_deg=0, phi_steer_deg=0):
    """
    Calculate array pattern for uniform planar array
    """
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    theta_steer = np.deg2rad(theta_steer_deg)
    phi_steer = np.deg2rad(phi_steer_deg)
    
    # Array factors for x and y dimensions
    psi_x = 2 * np.pi * dx_lambda * (np.sin(theta) * np.cos(phi) - np.sin(theta_steer) * np.cos(phi_steer))
    psi_y = 2 * np.pi * dy_lambda * (np.sin(theta) * np.sin(phi) - np.sin(theta_steer) * np.sin(phi_steer))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        AF_x = np.abs(np.sin(Nx * psi_x / 2) / (Nx * np.sin(psi_x / 2)))
        AF_x = np.where(np.isnan(AF_x), 1, AF_x)
        
        AF_y = np.abs(np.sin(Ny * psi_y / 2) / (Ny * np.sin(psi_y / 2)))
        AF_y = np.where(np.isnan(AF_y), 1, AF_y)
    
    AF = AF_x * AF_y
    
    return AF

def mimo_capacity(H, snr_db):
    """Calculate MIMO channel capacity"""
    snr_linear = 10**(snr_db/10)
    N_tx = H.shape[1]
    
    # Singular value decomposition
    _, S, _ = np.linalg.svd(H)
    
    # Water-filling power allocation (simplified: equal power)
    capacity = 0
    for s in S:
        capacity += np.log2(1 + snr_linear * s**2 / N_tx)
    
    return capacity

# ============================================================================
# 1. ANTENNA ARRAY PATTERNS
# ============================================================================
if app_mode == "📶 Antenna Array Patterns":
    st.markdown('<p class="section-header">📶 Antenna Array Patterns</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Array Configuration")
        
        array_type = st.radio("Array Type", ["Linear (ULA)", "Planar (UPA)"])
        
        if array_type == "Linear (ULA)":
            N_elements = st.slider("Number of Elements", 2, 32, 8)
            d_lambda = st.slider("Element Spacing (λ)", 0.1, 2.0, 0.5, 0.1)
            
            st.markdown("### Pattern Control")
            show_3db_beamwidth = st.checkbox("Show 3dB Beamwidth", value=True)
            show_sidelobes = st.checkbox("Highlight Sidelobes", value=True)
        
        else:  # Planar
            Nx = st.slider("Elements in X", 2, 16, 8)
            Ny = st.slider("Elements in Y", 2, 16, 8)
            d_lambda_planar = st.slider("Element Spacing (λ)", 0.1, 2.0, 0.5, 0.1, key='planar_d')
    
    with col2:
        if array_type == "Linear (ULA)":
            st.markdown("### Linear Array Pattern")
            
            # Calculate pattern
            theta_scan = np.linspace(-90, 90, 361)
            AF = uniform_linear_array_pattern(N_elements, d_lambda, theta_scan)
            AF_dB = 20 * np.log10(AF + 1e-10)
            
            # Create polar plot
            theta_rad = np.deg2rad(theta_scan)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=AF_dB,
                theta=theta_scan,
                mode='lines',
                name='Array Pattern',
                line=dict(color='blue', width=2)
            ))
            
            # Mark 3dB beamwidth
            if show_3db_beamwidth:
                idx_3db = np.where(AF_dB >= -3)[0]
                if len(idx_3db) > 0:
                    theta_3db_low = theta_scan[idx_3db[0]]
                    theta_3db_high = theta_scan[idx_3db[-1]]
                    beamwidth_3db = theta_3db_high - theta_3db_low
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[-3, -3],
                        theta=[theta_3db_low, theta_3db_high],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        name=f'3dB BW = {beamwidth_3db:.1f}°'
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        range=[-30, 0],
                        tickmode='linear',
                        tick0=0,
                        dtick=10
                    ),
                    angularaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=30
                    )
                ),
                title=f"Linear Array Pattern ({N_elements} elements, d={d_lambda}λ)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate key metrics
            main_lobe_width = beamwidth_3db if show_3db_beamwidth else 0
            
            # Find first null
            nulls = np.where(np.diff(np.sign(AF)))[0]
            first_null_angle = theta_scan[nulls[1]] if len(nulls) > 1 else 0
            
            # Directivity approximation
            directivity_db = 10 * np.log10(N_elements)
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Elements", N_elements)
            with col_b:
                st.metric("3dB Beamwidth", f"{main_lobe_width:.1f}°")
            with col_c:
                st.metric("First Null", f"{first_null_angle:.1f}°")
            with col_d:
                st.metric("Directivity", f"{directivity_db:.1f} dB")
        
        else:  # Planar array
            st.markdown("### Planar Array Pattern (3D)")
            
            # Create 3D pattern
            theta_3d = np.linspace(0, 90, 91)
            phi_3d = np.linspace(0, 360, 181)
            THETA, PHI = np.meshgrid(theta_3d, phi_3d)
            
            AF_3d = uniform_planar_array_pattern(Nx, Ny, d_lambda_planar, d_lambda_planar, 
                                                 THETA, PHI)
            AF_3d_dB = 20 * np.log10(AF_3d + 1e-10)
            
            # Convert to Cartesian for 3D plot
            X = AF_3d_dB * np.sin(np.deg2rad(THETA)) * np.cos(np.deg2rad(PHI))
            Y = AF_3d_dB * np.sin(np.deg2rad(THETA)) * np.sin(np.deg2rad(PHI))
            Z = AF_3d_dB * np.cos(np.deg2rad(THETA))
            
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
            
            fig.update_layout(
                title=f"Planar Array 3D Pattern ({Nx}x{Ny} elements)",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            total_elements = Nx * Ny
            directivity_planar = 10 * np.log10(total_elements)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Elements", total_elements)
            with col_b:
                st.metric("Array Size", f"{Nx} × {Ny}")
            with col_c:
                st.metric("Directivity", f"{directivity_planar:.1f} dB")
    
    with st.expander("📚 Theory: Antenna Arrays"):
        st.markdown("""
        ### Array Factor
        
        **Single antenna:** Radiates in all directions (omnidirectional or simple pattern)
        
        **Array of antennas:** Can focus energy in specific directions through constructive/destructive interference
        
        ### Uniform Linear Array (ULA)
        
        **Array Factor:**
        $$AF(\\theta) = \\left| \\frac{\\sin(N\\psi/2)}{N\\sin(\\psi/2)} \\right|$$
        
        Where:
        $$\\psi = 2\\pi \\frac{d}{\\lambda}(\\cos\\theta - \\cos\\theta_0)$$
        
        - $N$: Number of elements
        - $d$: Element spacing
        - $\\lambda$: Wavelength
        - $\\theta$: Observation angle
        - $\\theta_0$: Steering angle
        
        ### Key Parameters
        
        **Beamwidth (3dB):**
        $$BW_{3dB} \\approx \\frac{0.886\\lambda}{Nd\\cos\\theta_0}$$
        
        For broadside ($\\theta_0 = 90°$): $BW_{3dB} \\approx \\frac{50.8°}{N}$ (for $d=\\lambda/2$)
        
        **Directivity:**
        $$D \\approx N \\quad (\\text{for } d = \\lambda/2)$$
        
        In dB: $D_{dB} = 10\\log_{10}(N)$
        
        **Example:** 8-element array → 9 dB directivity gain
        
        ### Element Spacing Effects
        
        **d < λ/2:**
        - No grating lobes
        - Wider beamwidth
        - Lower directivity
        
        **d = λ/2:**
        - Optimal for most applications
        - No grating lobes for all steering angles
        - Standard choice
        
        **d > λ:**
        - Grating lobes appear (multiple main beams)
        - Narrower beamwidth
        - Undesirable in most cases
        
        ### Sidelobes
        
        **First sidelobe:** Typically -13.2 dB for uniform amplitude
        
        **Sidelobe reduction techniques:**
        - **Tapering (windowing):** Non-uniform amplitude distribution
          - Reduces sidelobe level
          - Increases beamwidth
          - Common windows: Hamming, Chebyshev, Taylor
        
        ### Planar Arrays
        
        **2D array** = Product of two linear arrays:
        $$AF(\\theta, \\phi) = AF_x(\\theta, \\phi) \\times AF_y(\\theta, \\phi)$$
        
        **Benefits:**
        - Narrower beamwidth in both dimensions
        - Higher directivity: $D \\approx N_x \\times N_y$
        - 2D beam steering
        - Used in 5G massive MIMO and mmWave
        
        ### 5G Applications
        
        **Sub-6 GHz:**
        - 8x8, 16x16 arrays typical
        - Analog/hybrid beamforming
        - Moderate directivity (15-25 dB)
        
        **mmWave:**
        - 64x64, 256 element arrays and larger
        - Essential for link budget (high path loss)
        - Pencil beams (<10° beamwidth)
        - High directivity (30-40 dB)
        """)

# ============================================================================
# 2. DIGITAL BEAMFORMING
# ============================================================================
elif app_mode == "🎯 Digital Beamforming":
    st.markdown('<p class="section-header">🎯 Digital Beamforming</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Beamforming Configuration")
        
        bf_type = st.radio("Beamforming Type", ["Analog", "Digital", "Hybrid"])
        
        N_ant = st.slider("Number of Antennas", 4, 64, 16)
        
        st.markdown("### Target Users")
        num_users = st.slider("Number of Users", 1, 8, 3)
        
        user_angles = []
        user_snrs = []
        
        for i in range(num_users):
            col_a, col_b = st.columns(2)
            with col_a:
                angle = st.slider(f"User {i+1} Angle (°)", -90, 90, -60 + i*40, key=f'angle_{i}')
                user_angles.append(angle)
            with col_b:
                snr = st.slider(f"User {i+1} SNR (dB)", -10, 30, 10, key=f'snr_{i}')
                user_snrs.append(snr)
    
    with col2:
        st.markdown("### Beamforming Pattern")
        
        # Calculate array patterns for each user
        theta_scan = np.linspace(-90, 90, 361)
        
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (angle, snr) in enumerate(zip(user_angles, user_snrs)):
            # Create beam steered to this user
            AF = uniform_linear_array_pattern(N_ant, 0.5, theta_scan, angle)
            AF_dB = 20 * np.log10(AF + 1e-10)
            
            fig.add_trace(go.Scatterpolar(
                r=AF_dB,
                theta=theta_scan,
                mode='lines',
                name=f'User {i+1} Beam ({angle}°)',
                line=dict(color=colors[i], width=2)
            ))
            
            # Mark user location
            fig.add_trace(go.Scatterpolar(
                r=[0],
                theta=[angle],
                mode='markers',
                marker=dict(size=15, color=colors[i], symbol='star'),
                name=f'User {i+1} Location',
                showlegend=False
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[-40, 0],
                    tickmode='linear',
                    tick0=0,
                    dtick=10
                ),
                angularaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=30
                )
            ),
            title=f"{bf_type} Beamforming - {N_ant} Antennas, {num_users} Users",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Capacity analysis
    st.markdown("### Multi-User MIMO Capacity")
    
    # Simplified capacity calculation
    total_capacity = 0
    user_capacities = []
    
    for i, (angle, snr) in enumerate(zip(user_angles, user_snrs)):
        # Beamforming gain (approximation)
        bf_gain_db = 10 * np.log10(N_ant)  # Ideal array gain (upper bound)
        effective_snr = snr + bf_gain_db
        
        # Shannon capacity
        capacity = np.log2(1 + 10**(effective_snr/10))
        user_capacities.append(capacity)
        total_capacity += capacity
    
    # Display capacities
    capacity_df = pd.DataFrame({
        'User': [f'User {i+1}' for i in range(num_users)],
        'Angle (°)': user_angles,
        'SNR (dB)': user_snrs,
        'Capacity (bits/s/Hz)': [f'{c:.2f}' for c in user_capacities]
    })
    
    st.dataframe(capacity_df, use_container_width=True)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Capacity", f"{total_capacity:.2f} bits/s/Hz")
    with col_b:
        st.metric("Avg per User", f"{total_capacity/num_users:.2f} bits/s/Hz")
    with col_c:
        st.metric("Ideal BF Gain", f"{10*np.log10(N_ant):.1f} dB")
    
    with st.expander("📚 Theory: Beamforming Types"):
        st.markdown("""
        ### Beamforming Fundamentals
        
        **Beamforming:** Technique to direct signal energy toward specific users/directions
        
        **Basic principle:** Phase shifts at array elements create constructive interference in desired direction
        
        ### Analog Beamforming
        
        **Implementation:**
        - Phase shifters in RF domain
        - One beam per RF chain
        - Low cost, low power
        
        **Characteristics:**
        - **Pros:** Simple hardware, energy efficient
        - **Cons:** Single beam (can't serve multiple users simultaneously in frequency)
        
        **Use case:** mmWave single-user scenarios
        
        ### Digital Beamforming
        
        **Implementation:**
        - One RF chain per antenna
        - Baseband processing for beamforming
        - Full flexibility
        
        **Characteristics:**
        - **Pros:** 
          - Multiple simultaneous beams
          - Adaptive per subcarrier/user
          - Multi-user MIMO
          - Best performance
        - **Cons:** High hardware cost and power
        
        **Use case:** Sub-6 GHz massive MIMO base stations
        
        ### Hybrid Beamforming
        
        **Implementation:**
        - Combines analog and digital
        - Analog for coarse beaming
        - Digital for fine tuning
        - N_RF < N_ant (fewer RF chains than antennas)
        
        **Characteristics:**
        - **Pros:** 
          - Balance between performance and cost
          - Multiple beams with reduced hardware
          - Practical for large arrays
        - **Cons:** More complex design
        
        **Use case:** mmWave 5G deployments
        
        ### Beamforming Weight Vector
        
        For user at angle $\\theta$:
        $$\\mathbf{w} = [1, e^{j\\psi}, e^{j2\\psi}, ..., e^{j(N-1)\\psi}]^T$$
        
        Where $\\psi = 2\\pi \\frac{d}{\\lambda}\\cos\\theta$
        
        **Received signal:**
        $$y = \\mathbf{w}^H \\mathbf{h} x + \\mathbf{w}^H \\mathbf{n}$$
        
        - $\\mathbf{h}$: Channel vector
        - $x$: Transmitted signal
        - $\\mathbf{n}$: Noise vector
        
        ### Multi-User Beamforming
        
        **Goal:** Serve K users simultaneously
        
        **Zero-Forcing (ZF):**
        $$\\mathbf{W} = \\mathbf{H}^H(\\mathbf{H}\\mathbf{H}^H)^{-1}$$
        
        Forces zero interference between users
        
        **MMSE (Minimum Mean Square Error):**
        $$\\mathbf{W} = \\mathbf{H}^H(\\mathbf{H}\\mathbf{H}^H + \\sigma^2\\mathbf{I})^{-1}$$
        
        Balances interference nulling with noise amplification
        
        **Maximum Ratio Transmission (MRT):**
        $$\\mathbf{w}_k = \\mathbf{h}_k^* / ||\\mathbf{h}_k||$$
        
        Maximizes SNR for each user independently
        
        ### Massive MIMO Beamforming
        
        When $N_{ant} >> N_{users}$:
        - Channel vectors become nearly orthogonal
        - Simple MRT performs well (near-optimal)
        - Interference naturally suppressed
        - "Channel hardening" effect
        
        **Favorable propagation:**
        $$\\mathbf{H}\\mathbf{H}^H \\approx \\alpha \\mathbf{I}$$
        
        ### 5G Beamforming
        
        **Codebook-based:**
        - Predefined set of beamforming vectors
        - UE measures and reports best beam
        - Lower overhead, suboptimal performance
        
        **CSI-based:**
        - UE reports detailed channel state
        - gNB computes optimal weights
        - Better performance, higher overhead
        
        **Beam sweeping:**
        - Try different beams sequentially
        - UE finds best beam
        - Used in initial access and beam management
        """)

# ============================================================================
# 3. BEAM STEERING
# ============================================================================
elif app_mode == "📡 Beam Steering":
    st.markdown('<p class="section-header">📡 Beam Steering</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Beam steering** dynamically changes the beam direction by adjusting antenna phase shifts.
    Critical for tracking moving users and mmWave communications.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Array Configuration")
        
        N_steer = st.slider("Number of Elements", 4, 32, 16, key='steer_n')
        d_steer = st.slider("Element Spacing (λ)", 0.3, 1.0, 0.5, 0.05, key='steer_d')
        
        st.markdown("### Steering Control")
        
        steer_angle = st.slider("Steering Angle (°)", -90, 90, 0, 5)
        
        animate = st.checkbox("Animate Beam Steering", value=False)
        
        if not animate:
            show_phase_shifts = st.checkbox("Show Phase Shifts", value=True)
    
    with col2:
        if not animate:
            st.markdown("### Steered Beam Pattern")
            
            theta_scan = np.linspace(-90, 90, 361)
            AF = uniform_linear_array_pattern(N_steer, d_steer, theta_scan, steer_angle)
            AF_dB = 20 * np.log10(AF + 1e-10)
            
            # Polar plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=AF_dB,
                theta=theta_scan,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Beam Pattern'
            ))
            
            # Mark steering direction
            fig.add_trace(go.Scatterpolar(
                r=[0],
                theta=[steer_angle],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name=f'Beam Direction ({steer_angle}°)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(range=[-30, 0]),
                    angularaxis=dict(tickmode='linear', tick0=0, dtick=30)
                ),
                title=f"Beam Steered to {steer_angle}°",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if show_phase_shifts:
                st.markdown("### Antenna Element Phase Shifts")
                
                # Calculate required phase shifts
                elements = np.arange(N_steer)
                phase_shifts = 2 * np.pi * d_steer * elements * np.cos(np.deg2rad(steer_angle))
                phase_shifts_deg = np.rad2deg(phase_shifts) % 360
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Bar(
                    x=elements,
                    y=phase_shifts_deg,
                    marker=dict(color=phase_shifts_deg, colorscale='Viridis'),
                    showlegend=False
                ))
                
                fig2.update_layout(
                    title=f"Phase Shifts for Steering to {steer_angle}°",
                    xaxis_title="Element Index",
                    yaxis_title="Phase Shift (degrees)",
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        else:
            st.markdown("### Animated Beam Steering")
            
            # Create animation frames
            angles_anim = np.linspace(-60, 60, 25)
            
            frames = []
            
            for angle in angles_anim:
                theta_scan = np.linspace(-90, 90, 361)
                AF = uniform_linear_array_pattern(N_steer, d_steer, theta_scan, angle)
                AF_dB = 20 * np.log10(AF + 1e-10)
                
                frame = go.Frame(
                    data=[
                        go.Scatterpolar(
                            r=AF_dB,
                            theta=theta_scan,
                            mode='lines',
                            line=dict(color='blue', width=2)
                        ),
                        go.Scatterpolar(
                            r=[0],
                            theta=[angle],
                            mode='markers',
                            marker=dict(size=15, color='red', symbol='star')
                        )
                    ],
                    name=str(angle)
                )
                frames.append(frame)
            
            # Initial frame
            theta_scan = np.linspace(-90, 90, 361)
            AF_init = uniform_linear_array_pattern(N_steer, d_steer, theta_scan, 0)
            AF_dB_init = 20 * np.log10(AF_init + 1e-10)
            
            fig = go.Figure(
                data=[
                    go.Scatterpolar(
                        r=AF_dB_init,
                        theta=theta_scan,
                        mode='lines',
                        line=dict(color='blue', width=2)
                    ),
                    go.Scatterpolar(
                        r=[0],
                        theta=[0],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star')
                    )
                ],
                layout=go.Layout(
                    polar=dict(
                        radialaxis=dict(range=[-30, 0]),
                        angularaxis=dict(tickmode='linear', tick0=0, dtick=30)
                    ),
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': 'Play',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 100, 'redraw': True},
                                    'fromcurrent': True,
                                    'mode': 'immediate'
                                }]
                            },
                            {
                                'label': 'Pause',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate'
                                }]
                            }
                        ]
                    }]
                ),
                frames=frames
            )
            
            fig.update_layout(height=600, title="Beam Steering Animation")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Steering range and speed
    st.markdown("### Beam Steering Capabilities")
    
    if d_steer <= 0.5:
        max_steer_angle = 90
    else:
        max_steer_angle = np.rad2deg(np.arcsin(1 / (2 * d_steer)))
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Max Steering Angle", f"±{max_steer_angle:.1f}°")
    with col_b:
        beamwidth = 50.8 / N_steer  # Approximate for d=λ/2
        st.metric("Beamwidth", f"{beamwidth:.1f}°")
    with col_c:
        num_beams = int(180 / beamwidth)
        st.metric("Beam Positions", f"~{num_beams}")
    
    with st.expander("📚 Theory: Beam Steering"):
        st.markdown("""
        ### Beam Steering Principle
        
        **Goal:** Point the main beam in direction $\\theta_0$
        
        **Method:** Apply progressive phase shift across array elements
        
        **Phase shift for element n:**
        $$\\phi_n = 2\\pi \\frac{d}{\\lambda} n \\cos\\theta_0$$
        
        Where:
        - $d$: Element spacing
        - $\\lambda$: Wavelength
        - $n$: Element index (0, 1, 2, ...)
        - $\\theta_0$: Desired beam direction
        
        ### Steering Range
        
        **Maximum steering angle:**
        $$\\theta_{max} = \\arcsin\\left(\\frac{\\lambda}{2d}\\right)$$
        
        **For d = λ/2:** Can steer to ±90° (entire hemisphere)
        **For d > λ:** Limited steering range, risk of grating lobes
        
        ### Grating Lobes
        
        **Appear when:** $d > \\lambda / (1 + |\\sin\\theta_0|)$
        
        **Consequence:** Multiple main beams (undesired)
        
        **Solution:** Keep $d \\leq \\lambda/2$ for all steering angles
        
        ### Beam Steering Speed
        
        **Electronic steering:**
        - **Analog:** Limited by phase shifter settling time (~μs)
        - **Digital:** Limited by processing time (~ms)
        
        **Mechanical steering:**
        - Motor-driven: Very slow (~seconds)
        - Not practical for mobile communications
        
        **5G requirement:**
        - Beam switching: <1 ms
        - Important for mobility and beam tracking
        
        ### Beam Squint
        
        **Problem:** In wideband systems, beam direction varies with frequency
        
        **Cause:** Phase shift is frequency-dependent
        $$\\phi(f) = 2\\pi \\frac{d}{c} f n \\cos\\theta_0$$
        
        **Impact:**
        - Beam points differently across OFDM subcarriers
        - Reduces beamforming gain
        - More severe for wide bandwidth and large arrays
        
        **Solution:**
        - True time delays (instead of phase shifters)
        - Hybrid beamforming
        - Frequency-dependent weights
        
        ### Beam Tracking
        
        **Challenge:** User moves → beam must follow
        
        **Approaches:**
        1. **Periodic beam sweeping:**
           - Re-measure all beams
           - High overhead, robust
        
        2. **Predictive tracking:**
           - Use mobility model
           - Estimate next position
           - Lower overhead, can fail
        
        3. **Hybrid:**
           - Track with narrow beam
           - Periodic verification with wider beam
           - Used in 5G NR
        
        ### 5G Beam Management
        
        **P-1 (Initial access):**
        - Coarse beam selection
        - gNB transmits SSB in different beams
        - UE measures and reports best beam
        
        **P-2 (Fine tuning):**
        - Refine beam pair
        - CSI-RS based measurements
        - Higher resolution
        
        **P-3 (Monitoring):**
        - Track current beam quality
        - Trigger beam change if needed
        
        **Beam failure recovery:**
        - Detect beam failure (RSRP drop)
        - Quickly find new beam
        - Critical for mmWave (blockage)
        """)

# ============================================================================
# 4. MIMO CAPACITY
# ============================================================================
elif app_mode == "🔢 MIMO Capacity":
    st.markdown('<p class="section-header">🔢 MIMO Channel Capacity</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### MIMO Configuration")
        
        N_tx = st.selectbox("TX Antennas", [1, 2, 4, 8, 16], index=2)
        N_rx = st.selectbox("RX Antennas", [1, 2, 4, 8, 16], index=2)
        
        st.markdown("### Channel Conditions")
        
        channel_type = st.radio("Channel Type", ["i.i.d. Rayleigh", "Correlated", "LOS (Rician)"])
        
        if channel_type == "Correlated":
            correlation = st.slider("Antenna Correlation", 0.0, 0.9, 0.3, 0.05)
        elif channel_type == "LOS (Rician)":
            K_factor = st.slider("Rician K-factor (dB)", -10, 20, 10)
        
        snr_range = st.slider("SNR Range (dB)", -10, 40, (-10, 30))
    
    # Generate channel realizations
    num_realizations = 1000
    snr_values = np.linspace(snr_range[0], snr_range[1], 50)
    
    capacities = np.zeros((len(snr_values), num_realizations))
    
    for idx, snr_db in enumerate(snr_values):
        for real in range(num_realizations):
            # Generate channel
            if channel_type == "i.i.d. Rayleigh":
                H = (np.random.randn(N_rx, N_tx) + 1j*np.random.randn(N_rx, N_tx)) / np.sqrt(2)
            elif channel_type == "Correlated":
                H_iid = (np.random.randn(N_rx, N_tx) + 1j*np.random.randn(N_rx, N_tx)) / np.sqrt(2)
                # Simple correlation model
                R = correlation * np.ones((N_tx, N_tx)) + (1-correlation) * np.eye(N_tx)
                H = H_iid @ scipy.linalg.sqrtm(R)
            else:  # Rician
                K_linear = 10**(K_factor/10)
                H_los = np.ones((N_rx, N_tx)) * np.sqrt(K_linear / (K_linear + 1))
                H_scatter = (np.random.randn(N_rx, N_tx) + 1j*np.random.randn(N_rx, N_tx)) / np.sqrt(2 * (K_linear + 1))
                H = H_los + H_scatter
            
            # Calculate capacity
            capacities[idx, real] = mimo_capacity(H, snr_db)
    
    # Average capacity
    avg_capacity = np.mean(capacities, axis=1)
    capacity_10th = np.percentile(capacities, 10, axis=1)
    capacity_90th = np.percentile(capacities, 90, axis=1)
    
    # SISO reference
    siso_capacity = np.log2(1 + 10**(snr_values/10))
    
    with col2:
        st.markdown("### MIMO Capacity vs SNR")
        
        fig = go.Figure()
        
        # SISO reference
        fig.add_trace(go.Scatter(
            x=snr_values,
            y=siso_capacity,
            mode='lines',
            name='SISO',
            line=dict(color='gray', dash='dash', width=2)
        ))
        
        # MIMO average
        fig.add_trace(go.Scatter(
            x=snr_values,
            y=avg_capacity,
            mode='lines',
            name=f'{N_tx}x{N_rx} MIMO (avg)',
            line=dict(color='blue', width=3)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([snr_values, snr_values[::-1]]),
            y=np.concatenate([capacity_90th, capacity_10th[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='10th-90th percentile',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"MIMO Capacity: {N_tx}x{N_rx} Configuration",
            xaxis_title="SNR (dB)",
            yaxis_title="Capacity (bits/s/Hz)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Capacity metrics
    st.markdown("### Capacity Metrics @ 20 dB SNR")
    
    idx_20db = np.argmin(np.abs(snr_values - 20))
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("SISO Capacity", f"{siso_capacity[idx_20db]:.2f} bits/s/Hz")
    with col_b:
        st.metric("MIMO Capacity", f"{avg_capacity[idx_20db]:.2f} bits/s/Hz")
    with col_c:
        mimo_gain = avg_capacity[idx_20db] / siso_capacity[idx_20db]
        st.metric("MIMO Gain", f"{mimo_gain:.2f}×")
    with col_d:
        theoretical_max = min(N_tx, N_rx) * siso_capacity[idx_20db]
        efficiency = avg_capacity[idx_20db] / theoretical_max * 100
        st.metric("Efficiency", f"{efficiency:.1f}%")
    
    with st.expander("📚 Theory: MIMO Capacity"):
        st.markdown("""
        ### MIMO Channel Model
        
        **Received signal:**
        $$\\mathbf{y} = \\mathbf{H}\\mathbf{x} + \\mathbf{n}$$
        
        Where:
        - $\\mathbf{y}$: $N_r \\times 1$ received vector
        - $\\mathbf{H}$: $N_r \\times N_t$ channel matrix
        - $\\mathbf{x}$: $N_t \\times 1$ transmitted vector
        - $\\mathbf{n}$: $N_r \\times 1$ noise vector
        
        ### MIMO Capacity
        
        **For known CSI at RX:**
        $$C = \\log_2 \\det\\left(\\mathbf{I}_{N_r} + \\frac{\\rho}{N_t}\\mathbf{H}\\mathbf{H}^H\\right)$$
        
        Where $\\rho$ is SNR.
        
        **SVD decomposition:**
        $$\\mathbf{H} = \\mathbf{U}\\mathbf{\\Sigma}\\mathbf{V}^H$$
        
        Capacity via parallel channels:
        $$C = \\sum_{i=1}^{r} \\log_2\\left(1 + \\frac{\\rho}{N_t}\\sigma_i^2\\right)$$
        
        Where:
        - $r = \\text{rank}(\\mathbf{H}) \\leq \\min(N_t, N_r)$
        - $\\sigma_i$ are singular values of $\\mathbf{H}$
        
        ### Key Insights
        
        **1. Multiplexing Gain:**
        - Capacity grows linearly with $\\min(N_t, N_r)$ at high SNR
        - Can send multiple independent streams
        - "Spatial multiplexing"
        
        **2. SISO vs MIMO:**
        - **SISO:** $C = \\log_2(1 + \\rho)$
        - **MIMO:** $C \\approx \\min(N_t, N_r) \\log_2(\\rho)$ at high SNR
        
        **Example (high SNR):**
        - 1x1: C ≈ 10 bits/s/Hz @ 30 dB
        - 4x4: C ≈ 40 bits/s/Hz @ 30 dB
        - **4× capacity gain!**
        
        **3. Diversity vs Multiplexing:**
        - **Diversity:** Use antennas for reliability
        - **Multiplexing:** Use antennas for capacity
        - Fundamental trade-off (can't maximize both)
        
        ### Channel Conditions Impact
        
        **i.i.d. Rayleigh (Rich scattering):**
        - Best case for MIMO
        - All paths independent
        - Full multiplexing gain achieved
        
        **Correlated Channel:**
        - When antennas are close or environment is sparse
        - Reduces effective degrees of freedom
        - Lower capacity than i.i.d.
        - Correlation $\\rho$: Higher correlation → lower capacity
        
        **LOS (Rician):**
        - Strong direct path + scattered components
        - Reduces rank of channel matrix
        - Lower multiplexing gain
        - Higher K-factor → approaches single path → lower capacity
        
        ### Water-Filling Power Allocation
        
        **Optimal power allocation** across spatial streams:
        
        For stream $i$:
        $$P_i = \\left(\\mu - \\frac{N_t}{\\rho \\sigma_i^2}\\right)^+$$
        
        Where $\\mu$ is chosen to satisfy $\\sum P_i = P_{total}$
        
        **Principle:**
        - Allocate more power to stronger channels (higher $\\sigma_i$)
        - Don't use weak channels (when $\\sigma_i$ too small)
        
        ### Practical MIMO Systems
        
        **LTE:**
        - Up to 4x4 MIMO
        - Typical: 2x2 or 4x2
        - Codebook-based precoding
        
        **5G NR:**
        - Up to 8 layers (8x8 effective)
        - Massive MIMO: 64, 128, 256 antennas at gNB
        - Type I/II codebooks
        - CSI feedback enhancements
        
        **Capacity in practice:**
        - Never reaches theoretical (imperfect CSI, interference)
        - Real-world: 50-70% of theoretical capacity
        - Still provides massive gains over SISO
        """)

# ============================================================================
# 5. MASSIVE MIMO
# ============================================================================
elif app_mode == "🌐 Massive MIMO":
    st.markdown('<p class="section-header">🌐 Massive MIMO</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Massive MIMO:** Large antenna arrays (64-256+ elements) at base station.
    Key technology for 5G capacity and coverage improvements.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Massive MIMO Configuration")
        
        M_bs = st.slider("BS Antennas", 16, 256, 64, 16)
        K_users = st.slider("Simultaneous Users", 1, 32, 8)
        
        st.markdown("### System Parameters")
        
        snr_massive = st.slider("Average SNR per User (dB)", -10, 30, 10, key='massive_snr')
        
        pilot_overhead = st.slider("Pilot Overhead (%)", 5, 30, 10)
        
        st.markdown("### Channel Model")
        user_spacing_deg = st.slider("Angular Separation (degrees)", 5, 30, 15)
    
    # Calculate massive MIMO performance
    snr_linear = 10**(snr_massive/10)
    
    # Array gain
    array_gain_db = 10 * np.log10(M_bs)
    
    # Effective SNR with array gain
    effective_snr_db = snr_massive + array_gain_db
    effective_snr_linear = 10**(effective_snr_db/10)
    
    # Per-user capacity (simplified)
    capacity_per_user = np.log2(1 + effective_snr_linear / K_users)
    
    # Sum capacity
    sum_capacity = K_users * capacity_per_user
    
    # Spectral efficiency (accounting for pilot overhead)
    spectral_efficiency = sum_capacity * (1 - pilot_overhead/100)
    
    with col2:
        st.markdown("### Massive MIMO Array Visualization")
        
        # Visualize antenna array
        array_config = int(np.sqrt(M_bs))
        if array_config**2 != M_bs:
            array_config = int(np.ceil(np.sqrt(M_bs)))
        
        x_pos = []
        y_pos = []
        
        for i in range(array_config):
            for j in range(array_config):
                if len(x_pos) < M_bs:
                    x_pos.append(i)
                    y_pos.append(j)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers',
            marker=dict(size=10, color='blue', symbol='square'),
            name='Antenna Elements'
        ))
        
        # Add user directions
        user_angles = np.linspace(-60, 60, K_users)
        
        for k, angle in enumerate(user_angles):
            # Plot user direction
            beam_length = array_config * 1.2
            x_beam = beam_length * np.sin(np.deg2rad(angle))
            y_beam = beam_length * np.cos(np.deg2rad(angle))
            
            fig.add_trace(go.Scatter(
                x=[array_config/2, array_config/2 + x_beam],
                y=[array_config/2, array_config/2 + y_beam],
                mode='lines+markers',
                line=dict(color=f'rgba(255, {100+k*20}, 0, 0.6)', width=2),
                marker=dict(size=[0, 15], symbol='star'),
                name=f'User {k+1}'
            ))
        
        fig.update_layout(
            title=f"Massive MIMO Array ({M_bs} antennas) serving {K_users} users",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("### Performance Metrics")
    
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    
    with col_a:
        st.metric("BS Antennas", M_bs)
    with col_b:
        st.metric("Active Users", K_users)
    with col_c:
        st.metric("Array Gain", f"{array_gain_db:.1f} dB")
    with col_d:
        st.metric("Sum Capacity", f"{sum_capacity:.1f} bits/s/Hz")
    with col_e:
        st.metric("Spectral Eff.", f"{spectral_efficiency:.1f} bits/s/Hz")
    
    # Scaling analysis
    st.markdown("### Massive MIMO Scaling")
    
    M_range = np.array([16, 32, 64, 128, 256])
    
    sum_cap_vs_M = []
    per_user_cap_vs_M = []
    
    for M in M_range:
        ag = 10 * np.log10(M)
        eff_snr = 10**((snr_massive + ag)/10)
        cap_pu = np.log2(1 + eff_snr / K_users)
        sum_cap_vs_M.append(K_users * cap_pu)
        per_user_cap_vs_M.append(cap_pu)
    
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sum Capacity vs BS Antennas', 'Per-User Capacity vs BS Antennas')
    )
    
    fig2.add_trace(
        go.Scatter(x=M_range, y=sum_cap_vs_M, mode='lines+markers',
                  line=dict(color='blue', width=2),
                  marker=dict(size=10)),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(x=M_range, y=per_user_cap_vs_M, mode='lines+markers',
                  line=dict(color='red', width=2),
                  marker=dict(size=10)),
        row=1, col=2
    )
    
    fig2.update_xaxes(title_text="Number of BS Antennas (M)", row=1, col=1)
    fig2.update_xaxes(title_text="Number of BS Antennas (M)", row=1, col=2)
    fig2.update_yaxes(title_text="Capacity (bits/s/Hz)", row=1, col=1)
    fig2.update_yaxes(title_text="Capacity (bits/s/Hz)", row=1, col=2)
    
    fig2.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("📚 Theory: Massive MIMO"):
        st.markdown("""
        ### What is Massive MIMO?
        
        **Definition:** Large antenna arrays (M >> K) where:
        - M: Number of BS antennas (64-256+)
        - K: Number of users (typically 8-32)
        - Ratio M/K: 4-32 (much larger than conventional MIMO)
        
        ### Key Benefits
        
        **1. Array Gain:**
        $$G_{array} = M \\quad (10\\log_{10}M \\text{ in dB})$$
        
        - 64 antennas → 18 dB gain
        - 256 antennas → 24 dB gain
        - Improves link budget dramatically
        
        **2. Spatial Multiplexing:**
        - Serve K users simultaneously on same frequency
        - Each user gets dedicated spatial channel
        - Sum capacity ≈ K × single-user capacity
        
        **3. Energy Efficiency:**
        - Reduce transmit power per antenna
        - Total power constant, but focused via beamforming
        - 10-100× better energy efficiency
        
        **4. Channel Hardening:**
        $$\\lim_{M \\to \\infty} \\frac{||\\mathbf{h}||^2}{M} = \\text{const}$$
        
        - Channel becomes deterministic as M grows
        - Less fading, more predictable
        - Simpler scheduling and resource allocation
        
        **5. Favorable Propagation:**
        $$\\mathbf{H}^H\\mathbf{H} \\approx \\beta M \\mathbf{I}_K$$
        
        - User channels become orthogonal
        - Interference naturally suppressed
        - Simple linear precoding (MRT/MRC) near-optimal
        
        ### Precoding for Massive MIMO
        
        **Maximum Ratio Transmission (MRT):**
        $$\\mathbf{W} = \\mathbf{H}^H$$
        
        - Simplest, lowest complexity
        - Near-optimal for large M/K ratios
        - Used in practice
        
        **Zero-Forcing (ZF):**
        $$\\mathbf{W} = \\mathbf{H}^H(\\mathbf{H}\\mathbf{H}^H)^{-1}$$
        
        - Forces zero inter-user interference
        - Better for moderate M/K
        - Higher complexity
        
        ### TDD Reciprocity
        
        **Critical advantage of TDD + Massive MIMO:**
        
        $$\\mathbf{H}_{UL} = \\mathbf{H}_{DL}^T$$
        
        **Why important:**
        1. BS learns DL channel from UL pilots
        2. No DL CSI feedback needed
        3. Scales to any M (feedback would be M×K overhead!)
        4. Enables massive arrays
        
        **FDD challenges:**
        - Feedback grows with M
        - Not practical for M > 32
        - Why FDD massive MIMO is harder
        
        ### Pilot Contamination
        
        **Problem:** Finite orthogonal pilots in TDD
        
        - K users need K orthogonal pilots
        - Neighboring cells reuse pilots
        - "Pilot contamination": channels interfere during estimation
        
        **Impact:**
        - Limits achievable SINR
        - Does NOT disappear as M → ∞
        - Main bottleneck in massive MIMO
        
        **Solutions:**
        - Pilot assignment coordination
        - Blind channel estimation
        - Superimposed pilots
        
        ### Practical Considerations
        
        **Hardware:**
        - Need M RF chains (expensive!)
        - Power consumption
        - Physical size of array
        - Calibration (reciprocity only after calibration)
        
        **Signal processing:**
        - Matrix operations scale as O(M²K)
        - Need efficient implementations
        - Real-time processing challenges
        
        **Deployment:**
        - **sub-6 GHz:** 64-128 antennas typical
        - **mmWave:** 256+ antennas possible (smaller wavelength)
        
        ### 5G Massive MIMO
        
        **3GPP support:**
        - Up to 32 CSI-RS ports
        - Enhanced Type II codebooks
        - Beam management procedures
        
        **Real deployments:**
        - China Mobile: 64T64R (64 TX, 64 RX)
        - Sprint (now T-Mobile): 128-antenna arrays
        - Verizon mmWave: 256-antenna panels
        
        **Performance:**
        - 5-10× capacity vs 4x4 MIMO
        - Better coverage (array gain)
        - Improved edge-user throughput
        
        ### Capacity Formula
        
        **Uplink sum capacity (TDD, MRC):**
        $$C_{UL} = K \\log_2\\left(1 + \\frac{M \\rho}{K}\\right)$$
        
        As M/K → ∞:
        $$C_{UL} \\to K \\log_2\\left(\\frac{M \\rho}{K}\\right)$$
        
        Grows logarithmically with M, linearly with K.
        """)

# ============================================================================
# 6. BEAM MANAGEMENT
# ============================================================================
elif app_mode == "📊 Beam Management":
    st.markdown('<p class="section-header">📊 5G Beam Management</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Beam management** in 5G NR includes procedures for beam selection, refinement, and recovery.
    Critical for mmWave where beams are narrow and susceptible to blockage.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🔍 Beam Sweeping", "📈 Beam Refinement", "🔄 Beam Recovery"])
    
    with tab1:
        st.markdown("### P-1: Initial Beam Selection (Beam Sweeping)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            num_beams = st.slider("Number of SSB Beams", 4, 64, 8)
            sector_angle = st.slider("Sector Coverage (°)", 60, 120, 120)
            
            st.markdown("""
            **Procedure:**
            1. gNB transmits SSBs in different beam directions
            2. UE measures RSRP for each SSB beam
            3. UE reports best beam(s)
            4. Initial beam pair established
            """)
        
        with col2:
            # Visualize beam sweeping
            beam_angles = np.linspace(-sector_angle/2, sector_angle/2, num_beams)
            
            fig = go.Figure()
            
            for idx, angle in enumerate(beam_angles):
                theta = np.linspace(angle - sector_angle/(2*num_beams), 
                                  angle + sector_angle/(2*num_beams), 50)
                r = np.ones_like(theta)
                
                fig.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta,
                    mode='lines',
                    fill='toself',
                    name=f'Beam {idx}',
                    fillcolor=f'rgba({50+idx*200//num_beams}, 100, 200, 0.3)',
                    line=dict(color='blue', width=1)
                ))
            
            # Mark UE position
            ue_angle = 25  # Example
            fig.add_trace(go.Scatterpolar(
                r=[1.2],
                theta=[ue_angle],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='UE Location'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=False)),
                title=f"SSB Beam Sweeping ({num_beams} beams)",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### P-2: Beam Refinement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Purpose:** Fine-tune beam pair for better SINR
            
            **Mechanism:**
            - CSI-RS based measurements
            - Higher angular resolution than SSB
            - Both TX and RX beams can be refined
            
            **Refinement levels:**
            1. **Coarse (P-1):** ~10-30° beamwidth
            2. **Medium:** ~5-10° beamwidth  
            3. **Fine (P-2):** ~2-5° beamwidth
            """)
        
        with col2:
            # Show refinement process
            refinement_data = pd.DataFrame({
                'Stage': ['P-1 (Coarse)', 'P-2 (Medium)', 'P-2 (Fine)'],
                'Beamwidth (°)': [20, 8, 3],
                'Gain (dB)': [15, 21, 26],
                'Measurement': ['SSB RSRP', 'CSI-RS', 'CSI-RS']
            })
            
            st.dataframe(refinement_data, use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=refinement_data['Stage'],
                y=refinement_data['Gain (dB)'],
                marker=dict(color=['lightblue', 'blue', 'darkblue']),
                text=refinement_data['Gain (dB)'],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Beamforming Gain by Refinement Stage",
                yaxis_title="Gain (dB)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### P-3: Beam Failure Recovery")
        
        st.markdown("""
        **Beam failure** occurs when:
        - Blockage (person, vehicle, building)
        - UE rotation (phone tilted)
        - Mobility (moved out of beam)
        
        **Detection:**
        - Monitor beam quality (L1-RSRP)
        - Declare failure if below threshold for N consecutive measurements
        
        **Recovery procedure:**
        1. **Trigger:** UE detects beam failure
        2. **New beam search:** UE tries to find alternative beam
        3. **Request:** Send PRACH on new beam candidate
        4. **Response:** gNB acknowledges and switches beam
        5. **Confirmation:** Resume data transmission
        """)
        
        # Beam failure timeline
        col1, col2 = st.columns([1, 2])
        
        with col1:
            bfr_latency = st.slider("BFR Target Latency (ms)", 10, 100, 30)
            detection_time = bfr_latency * 0.4
            search_time = bfr_latency * 0.3
            recovery_time = bfr_latency * 0.3
        
        with col2:
            # Timeline visualization
            stages = ['Detection', 'Search', 'Recovery']
            times = [detection_time, search_time, recovery_time]
            
            fig = go.Figure()
            
            cumulative_time = 0
            for stage, time in zip(stages, times):
                fig.add_trace(go.Bar(
                    x=[time],
                    y=[stage],
                    orientation='h',
                    name=stage,
                    text=f'{time:.1f} ms',
                    textposition='inside'
                ))
                cumulative_time += time
            
            fig.update_layout(
                title=f"Beam Failure Recovery Timeline (Total: {bfr_latency} ms)",
                xaxis_title="Time (ms)",
                height=300,
                showlegend=False,
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Impact on Service:**
        - Voice call: {bfr_latency}ms interruption (acceptable if < 50ms)
        - Video stream: Buffer can absorb short outage
        - URLLC: May violate latency requirements
        
        **Mitigation:**
        - Multi-beam transmission
        - Beam diversity
        - Faster BFR procedures
        """)
    
    with st.expander("📚 Theory: Beam Management"):
        st.markdown("""
        ### 5G NR Beam Management Framework
        
        **Goal:** Establish and maintain optimal beam pair between gNB and UE
        
        ### P-1: Beam Sweeping (Initial Access)
        
        **SSB (SS/PBCH Block) transmission:**
        - Multiple beams in time-division manner
        - Each beam carries same info (PSS/SSS/PBCH)
        - Beam index implicitly indicated
        
        **Configuration:**
        - **FR1 (<6 GHz):** Up to 8 SSB beams
        - **FR2 (mmWave):** Up to 64 SSB beams
        
        **UE procedure:**
        1. Measure L1-RSRP for each SSB
        2. Select best beam(s)
        3. Report via PRACH or RRC
        
        **Periodicity:** 5-160 ms (configurable)
        
        ### P-2: Beam Refinement
        
        **CSI-RS for beam management:**
        - More flexible than SSB
        - Can be beamformed
        - Aperiodic, semi-persistent, or periodic
        
        **Two-stage refinement:**
        1. **TX beam refinement:** gNB tries different TX beams, UE measures
        2. **RX beam refinement:** UE tries different RX beams, reports best
        
        **Beam reporting:**
        - **L1-RSRP:** Fast, low resolution (2-4 bits)
        - **L3-RSRP:** Slower, higher resolution (7 bits)
        
        ### P-3: Beam Failure Recovery (BFR)
        
        **Beam failure declaration:**
        - Monitor hypothetical PDCCH BLER
        - If BLER > threshold for N instances → beam failure
        
        **Candidate beam identification:**
        - **Periodic:** UE monitors alternate beams continuously
        - **New beam search:** After failure, search for new beam
        
        **Recovery request:**
        - PRACH-based (contention or non-contention)
        - Or PUCCH (if available)
        
        **Timeline:**
        - Detection: 5-20 ms
        - Search: 5-10 ms
        - Request + Response: 10-20 ms
        - **Total:** 20-50 ms typical
        
        ### Multi-Beam Operation
        
        **Beam correspondence:**
        - Assumption: Best RX beam corresponds to best TX beam
        - Valid in reciprocal channels (TDD)
        - Reduces overhead
        
        **Beam sweeping patterns:**
        - **Sequential:** One beam at a time
        - **Simultaneous:** Multiple beams (if enough RF chains)
        
        ### Beam Measurement and Reporting
        
        **Measurements:**
        - **L1-RSRP:** Fast, based on CSI-RS
        - **L1-SINR:** Includes interference
        
        **Reporting:**
        - **CRI (CSI-RS Resource Indicator):** Which CSI-RS resource
        - **SSBRI (SSB Resource Indicator):** Which SSB
        - Can report top-K beams
        
        ### Mobility and Beam Tracking
        
        **Challenge:** User moves → beam may no longer be optimal
        
        **Approaches:**
        1. **Periodic re-measurement:** Regular P-2 procedure
        2. **Threshold-based:** Trigger when quality degrades
        3. **Predictive:** Use mobility model
        
        **Handover:**
        - L3-mobility (RRC) for cell changes
        - L1/L2-mobility for intra-cell beam changes
        
        ### mmWave Specific Challenges
        
        **Blockage:**
        - Human body: 20-30 dB attenuation
        - Vehicles: 10-20 dB
        - Buildings: Complete blockage
        
        **Solution:**
        - Multi-connectivity (anchor + booster)
        - Fast BFR (<10 ms)
        - Spatial diversity
        
        **Beam alignment:**
        - Narrow beams (<10°) require precise alignment
        - Sensitive to orientation changes
        - Accelerometer-aided beam tracking
        
        ### 3GPP Procedures
        
        **Beam management procedures defined in 3GPP:**
        - P-1: 38.213 Section 5.1
        - P-2/P-3: 38.214 Section 5.2
        - BFR: 38.213 Section 5.17
        
        **RS for beam management:**
        - SSB (always on)
        - CSI-RS for beam management (configurable)
        - CSI-RS for beam failure detection
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Beamforming & MIMO Simulator</strong> | Built with Streamlit</p>
    <p>Explore antenna arrays, beamforming, and massive MIMO systems</p>
</div>
""", unsafe_allow_html=True)

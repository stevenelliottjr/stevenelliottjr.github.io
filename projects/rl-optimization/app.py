import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from rl_optimizer import ResourceOptimizer, ResourceAllocationEnv
import os

# Set page configuration
st.set_page_config(
    page_title="RL for Resource Optimization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background-color: #d1e7dd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stPlotlyChart {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def create_priority_chart(priorities):
    """Create an Altair chart to visualize task priorities"""
    data = pd.DataFrame({
        'Task': [f"Task {i+1}" for i in range(len(priorities))],
        'Priority': priorities
    })
    
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Priority:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Task:N', sort='-x'),
        color=alt.Color('Priority:Q', scale=alt.Scale(scheme='viridis'))
    ).properties(
        width=400,
        height=300,
        title='Task Priorities'
    )
    
    return chart

def create_allocation_chart(allocation, priorities=None, comparison=None):
    """Create an Altair chart to visualize resource allocation"""
    if comparison is None:
        # Single allocation visualization
        data = pd.DataFrame({
            'Task': [f"Task {i+1}" for i in range(len(allocation))],
            'Resources': allocation,
            'Priority': priorities if priorities is not None else np.ones(len(allocation))
        })
        
        base = alt.Chart(data).encode(
            y=alt.Y('Task:N', sort='-x', title=None)
        )
        
        bars = base.mark_bar().encode(
            x=alt.X('Resources:Q', title='Resources Allocated'),
            color=alt.Color('Priority:Q', scale=alt.Scale(scheme='viridis'))
        )
        
        text = base.mark_text(
            align='left',
            baseline='middle',
            dx=3
        ).encode(
            text=alt.Text('Resources:Q'),
            x=alt.X('Resources:Q')
        )
        
        chart = (bars + text).properties(
            width=500,
            height=300,
            title='Resource Allocation'
        )
    
    else:
        # Comparison visualization
        rl_allocation = allocation
        optimal_allocation = comparison["optimal_allocation"]
        
        data = pd.DataFrame({
            'Task': np.repeat([f"Task {i+1}" for i in range(len(rl_allocation))], 2),
            'Resources': np.concatenate([rl_allocation, optimal_allocation]),
            'Method': ['RL Agent'] * len(rl_allocation) + ['Optimal'] * len(optimal_allocation),
            'Priority': np.tile(comparison["task_priorities"], 2)
        })
        
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('Resources:Q', title='Resources Allocated'),
            y=alt.Y('Task:N', title=None),
            color=alt.Color('Method:N', scale=alt.Scale(domain=['RL Agent', 'Optimal'], 
                                                      range=['#2C7FB8', '#7FBC41'])),
            row=alt.Row('Method:N')
        ).properties(
            width=500,
            height=150,
            title=f'Allocation Comparison (RL Performance: {comparison["optimality_percentage"]:.1f}% of Optimal)'
        )
    
    return chart

def main():
    st.sidebar.image("https://raw.githubusercontent.com/stevenelliottjr/rl-optimization/main/images/logo.png", width=200)
    st.sidebar.title("⚙️ Configuration")
    
    # Main sections of the app
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["📊 Demo", "🧪 Experiment", "📚 Learn", "ℹ️ About"]
    )
    
    # Initialize session state
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'comparison' not in st.session_state:
        st.session_state.comparison = None
    
    # Demo Mode
    if app_mode == "📊 Demo":
        st.markdown('<p class="main-header">🤖 Reinforcement Learning for Resource Optimization</p>', unsafe_allow_html=True)
        
        st.markdown("""
        This interactive demo showcases how reinforcement learning can be applied to resource allocation problems. 
        The RL agent learns to optimally distribute limited resources across multiple tasks with different priorities.
        """)
        
        # Problem Configuration
        st.markdown('<p class="sub-header">Problem Configuration</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_resources = st.slider("Number of Resources", min_value=5, max_value=50, value=20, step=1,
                                     help="Total number of resources to allocate")
            
            num_tasks = st.slider("Number of Tasks", min_value=3, max_value=15, value=5, step=1,
                                 help="Number of tasks that need resources")
            
            max_per_task = st.slider("Maximum Resources Per Task", min_value=1, max_value=num_resources, 
                                    value=min(10, num_resources), step=1,
                                   help="Maximum resources that can be allocated to any single task")
            
            algorithm = st.selectbox("RL Algorithm", ["PPO", "DQN", "A2C"], index=0,
                                    help="Reinforcement learning algorithm to use")
        
        with col2:
            st.markdown('<div class="info-box">Task Priorities</div>', unsafe_allow_html=True)
            
            priority_mode = st.radio("Priority Mode", ["Random", "Custom"], index=0)
            
            if priority_mode == "Random":
                # Generate random priorities
                random_seed = st.slider("Random Seed", min_value=1, max_value=100, value=42, step=1)
                np.random.seed(random_seed)
                priorities = np.random.uniform(0.1, 1.0, size=num_tasks)
                priorities = priorities / np.sum(priorities)  # Normalize
                
                # Show the generated priorities
                st.write("Generated Task Priorities:")
                st.altair_chart(create_priority_chart(priorities), use_container_width=True)
                
            else:
                # Custom priorities with sliders
                priorities = []
                for i in range(num_tasks):
                    priority = st.slider(f"Priority for Task {i+1}", min_value=0.1, max_value=1.0, 
                                        value=round(0.5, 1), step=0.1)
                    priorities.append(priority)
                
                # Normalize priorities to sum to 1
                priorities = np.array(priorities) / np.sum(priorities)
                
                # Show the normalized priorities
                st.write("Normalized Task Priorities:")
                st.altair_chart(create_priority_chart(priorities), use_container_width=True)
        
        # Configure and train the agent
        constraints = {"max_per_task": max_per_task}
        
        if st.button("Train RL Agent", key="train_button"):
            with st.spinner("Training agent... This may take a moment."):
                try:
                    # Initialize the optimizer
                    optimizer = ResourceOptimizer(
                        resources=num_resources,
                        tasks=num_tasks,
                        task_priorities=priorities,
                        algorithm=algorithm.lower(),
                        constraints=constraints,
                        reward_type="weighted_completion"
                    )
                    
                    # Train the agent (with fewer episodes for demo)
                    optimizer.train(episodes=500, learning_rate=0.001, gamma=0.99, verbose=0)
                    
                    # Store in session state
                    st.session_state.optimizer = optimizer
                    st.session_state.trained = True
                    
                    # Generate a comparison
                    comparison = optimizer.compare_to_optimal()
                    st.session_state.comparison = comparison
                    
                    st.success("Agent trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        
        # Show results if the agent is trained
        if st.session_state.trained and st.session_state.optimizer is not None:
            st.markdown('<p class="sub-header">Optimization Results</p>', unsafe_allow_html=True)
            
            # Display comparison results
            comparison = st.session_state.comparison
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="success-box">
                <b>Performance:</b> The RL agent achieved {comparison['optimality_percentage']:.1f}% of the theoretical optimal allocation.
                </div>
                """, unsafe_allow_html=True)
                
                # Show allocation values
                comparison_df = pd.DataFrame({
                    'Task': [f"Task {i+1}" for i in range(len(comparison['rl_allocation']))],
                    'RL Allocation': comparison['rl_allocation'],
                    'Optimal Allocation': comparison['optimal_allocation'],
                    'Priority': comparison['task_priorities']
                })
                
                st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['RL Allocation', 'Optimal Allocation']), height=300)
            
            with col2:
                # Display allocation comparison chart
                allocation_chart = create_allocation_chart(
                    comparison['rl_allocation'], 
                    comparison=comparison
                )
                st.altair_chart(allocation_chart, use_container_width=True)
            
            # Additional experiments section
            st.markdown('<p class="sub-header">Try Different Scenarios</p>', unsafe_allow_html=True)
            
            st.write("Adjust resources or priorities to see how the trained agent adapts:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_resources = st.slider("Test Resources", min_value=5, max_value=50, 
                                          value=num_resources, step=1)
                
                # Custom test priorities
                st.write("Adjust task priorities:")
                test_priorities = []
                for i in range(num_tasks):
                    test_priority = st.slider(f"Test Priority {i+1}", min_value=0.1, max_value=1.0, 
                                            value=round(priorities[i], 1), step=0.1, key=f"test_pri_{i}")
                    test_priorities.append(test_priority)
                
                # Normalize test priorities
                test_priorities = np.array(test_priorities) / np.sum(test_priorities)
            
            with col2:
                if st.button("Run Test Scenario"):
                    with st.spinner("Generating allocation..."):
                        optimizer = st.session_state.optimizer
                        
                        # Get the new allocation
                        new_allocation = optimizer.allocate(
                            resources=test_resources,
                            task_priorities=test_priorities
                        )
                        
                        # Compare to optimal
                        new_comparison = optimizer.compare_to_optimal(
                            resources=test_resources,
                            task_priorities=test_priorities
                        )
                        
                        # Display the new comparison chart
                        new_allocation_chart = create_allocation_chart(
                            new_allocation, 
                            comparison=new_comparison
                        )
                        st.altair_chart(new_allocation_chart, use_container_width=True)
    
    # Experiment Mode
    elif app_mode == "🧪 Experiment":
        st.markdown('<p class="main-header">Experiment with RL Parameters</p>', unsafe_allow_html=True)
        
        st.markdown("""
        This section allows you to experiment with different RL algorithms and hyperparameters to see 
        how they affect the performance of resource allocation.
        """)
        
        # Problem setup
        st.markdown('<p class="sub-header">Problem Definition</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_resources = st.slider("Resources", min_value=5, max_value=50, value=15, step=1)
            experiment_tasks = st.slider("Tasks", min_value=3, max_value=10, value=5, step=1)
            experiment_max_per_task = st.slider("Max Per Task", min_value=1, max_value=experiment_resources, 
                                              value=min(8, experiment_resources), step=1)
            
            # Generate random priorities for experiment
            st.write("Random Task Priorities:")
            np.random.seed(42)  # Fixed seed for reproducibility
            experiment_priorities = np.random.uniform(0.1, 1.0, size=experiment_tasks)
            experiment_priorities = experiment_priorities / np.sum(experiment_priorities)
            
            st.altair_chart(create_priority_chart(experiment_priorities), use_container_width=True)
        
        with col2:
            st.markdown('<p class="sub-header">RL Parameters</p>', unsafe_allow_html=True)
            
            # Algorithm selection
            experiment_algorithm = st.selectbox("Algorithm", ["PPO", "DQN", "A2C"], index=0)
            
            # Hyperparameters
            experiment_episodes = st.slider("Training Episodes", min_value=100, max_value=2000, value=1000, step=100)
            experiment_learning_rate = st.select_slider("Learning Rate", 
                                                       options=[0.0001, 0.0003, 0.001, 0.003, 0.01], 
                                                       value=0.001)
            experiment_gamma = st.slider("Discount Factor (gamma)", min_value=0.8, max_value=0.99, value=0.99, step=0.01)
            
            # Reward function
            experiment_reward_type = st.selectbox("Reward Function", 
                                                ["weighted_completion", "balanced", "sparse"], 
                                                index=0,
                                               help="Different reward functions incentivize different allocation strategies")
        
        # Create experiment configuration
        experiment_constraints = {"max_per_task": experiment_max_per_task}
        
        # Run the experiment
        if st.button("Run Experiment", key="experiment_button"):
            with st.spinner(f"Running experiment with {experiment_algorithm}... This may take a while."):
                try:
                    # Initialize the optimizer with experiment settings
                    experiment_optimizer = ResourceOptimizer(
                        resources=experiment_resources,
                        tasks=experiment_tasks,
                        task_priorities=experiment_priorities,
                        algorithm=experiment_algorithm.lower(),
                        constraints=experiment_constraints,
                        reward_type=experiment_reward_type
                    )
                    
                    # Train the agent with specified parameters
                    experiment_optimizer.train(
                        episodes=experiment_episodes,
                        learning_rate=experiment_learning_rate,
                        gamma=experiment_gamma,
                        verbose=1
                    )
                    
                    # Generate a comparison
                    experiment_comparison = experiment_optimizer.compare_to_optimal()
                    
                    st.success(f"Experiment completed! Performance: {experiment_comparison['optimality_percentage']:.1f}% of optimal")
                    
                    # Display experiment results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Training history plot
                        st.markdown('<p class="sub-header">Training Progress</p>', unsafe_allow_html=True)
                        fig = experiment_optimizer.plot_training_history()
                        st.pyplot(fig)
                    
                    with col2:
                        # Allocation comparison
                        st.markdown('<p class="sub-header">Allocation Comparison</p>', unsafe_allow_html=True)
                        experiment_allocation_chart = create_allocation_chart(
                            experiment_comparison['rl_allocation'], 
                            comparison=experiment_comparison
                        )
                        st.altair_chart(experiment_allocation_chart, use_container_width=True)
                    
                    # Detailed results
                    st.markdown('<p class="sub-header">Detailed Results</p>', unsafe_allow_html=True)
                    
                    # Create a DataFrame for the results
                    results_df = pd.DataFrame({
                        'Task': [f"Task {i+1}" for i in range(experiment_tasks)],
                        'Priority': experiment_comparison['task_priorities'],
                        'RL Allocation': experiment_comparison['rl_allocation'],
                        'Optimal Allocation': experiment_comparison['optimal_allocation'],
                        'Difference': experiment_comparison['rl_allocation'] - experiment_comparison['optimal_allocation']
                    })
                    
                    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Priority'])
                               .highlight_min(axis=0, subset=['Difference']), height=300)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Optimality", f"{experiment_comparison['optimality_percentage']:.1f}%")
                    with col2:
                        st.metric("RL Value", f"{experiment_comparison['rl_weighted_value']:.2f}")
                    with col3:
                        st.metric("Optimal Value", f"{experiment_comparison['optimal_weighted_value']:.2f}")
                    with col4:
                        # Mean absolute difference between allocations
                        mae = np.mean(np.abs(experiment_comparison['rl_allocation'] - experiment_comparison['optimal_allocation']))
                        st.metric("Allocation MAE", f"{mae:.2f}")
                
                except Exception as e:
                    st.error(f"Error during experiment: {str(e)}")
    
    # Learn Mode
    elif app_mode == "📚 Learn":
        st.markdown('<p class="main-header">Learning Resources</p>', unsafe_allow_html=True)
        
        st.markdown("""
        Learn about reinforcement learning and how it can be applied to optimization problems.
        """)
        
        # Create tabs for different learning sections
        tab1, tab2, tab3 = st.tabs(["RL Basics", "Optimization with RL", "Mathematics"])
        
        with tab1:
            st.markdown("""
            ## Reinforcement Learning Basics
            
            Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize a reward signal. The key components of RL are:
            
            1. **Agent**: The entity that makes decisions and takes actions
            2. **Environment**: The world in which the agent operates
            3. **State**: The current situation of the agent in the environment
            4. **Action**: The moves available to the agent
            5. **Reward**: The feedback signal that indicates how well the agent is doing
            
            ### Key RL Algorithms
            
            This demo uses three popular RL algorithms:
            
            **Proximal Policy Optimization (PPO)**
            - A policy gradient method that performs well on a variety of tasks
            - Maintains a good balance between sample efficiency and ease of implementation
            - Uses a "clipped" objective function to limit policy updates
            
            **Deep Q-Network (DQN)**
            - Combines Q-learning with deep neural networks
            - Uses experience replay to break correlations in the observation sequence
            - Employs a separate target network to stabilize learning
            
            **Advantage Actor-Critic (A2C)**
            - Hybrid approach combining policy gradients and value-based methods
            - The actor determines the policy (how to act), while the critic evaluates the action
            - Reduces variance in updates while maintaining reasonable bias
            """)
            
            st.image("https://raw.githubusercontent.com/stevenelliottjr/rl-optimization/main/images/rl_diagram.png", caption="Reinforcement Learning Loop")
        
        with tab2:
            st.markdown("""
            ## Resource Allocation with RL
            
            Resource allocation is a common optimization problem where limited resources must be distributed across multiple tasks or projects. Traditional approaches include:
            
            - **Linear Programming**: Works well for simple constraints but struggles with complex, non-linear problems
            - **Genetic Algorithms**: Can handle complex problems but may be slow to converge
            - **Heuristic Methods**: Fast but may get stuck in local optima
            
            ### Why Use RL for Resource Allocation?
            
            Reinforcement learning offers several advantages:
            
            1. **Adaptability**: RL agents can adapt to changing conditions and learn from experience
            2. **Complex Constraints**: Can handle non-linear constraints and complex reward structures
            3. **Sequential Decision Making**: Models the problem as a series of decisions rather than a single optimization
            4. **Scalability**: Can scale to large problems with many variables
            
            ### How It Works in This Demo
            
            In our demo:
            - The **state** is the current allocation of resources to tasks
            - **Actions** are decisions about which task to allocate the next resource to
            - The **reward** is based on how well the allocation matches the priorities of tasks
            - The **environment** enforces constraints like maximum resources per task
            
            The agent learns through thousands of episodes, gradually improving its allocation strategy to maximize the weighted value of the allocation.
            """)
            
            st.image("https://raw.githubusercontent.com/stevenelliottjr/rl-optimization/main/images/allocation_example.png", caption="Example Resource Allocation")
        
        with tab3:
            st.markdown("""
            ## Mathematical Foundations
            
            ### Markov Decision Process (MDP)
            
            RL problems are typically formulated as MDPs, defined by the tuple (S, A, P, R, γ) where:
            - S is the set of states
            - A is the set of actions
            - P is the transition probability function, P(s'|s,a)
            - R is the reward function, R(s,a,s')
            - γ is the discount factor
            
            The goal is to find a policy π(a|s) that maximizes the expected cumulative discounted reward:
            
            $$ V^\\pi(s) = E_\\pi \\left[ \\sum_{t=0}^{\\infty} \\gamma^t R(s_t, a_t, s_{t+1}) \\mid s_0 = s \\right] $$
            
            ### Bellman Equation
            
            The optimal value function satisfies the Bellman optimality equation:
            
            $$ V^*(s) = \\max_a \\left[ R(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V^*(s') \\right] $$
            
            ### Policy Gradient Methods
            
            Policy gradient methods directly optimize the policy by estimating the gradient of the expected return:
            
            $$ \\nabla_\\theta J(\\theta) = E_{\\pi_\\theta} \\left[ \\nabla_\\theta \\log \\pi_\\theta(a|s) Q^{\\pi_\\theta}(s,a) \\right] $$
            
            ### Value Function Approximation
            
            For large state spaces, we use function approximation:
            
            $$ V(s) \\approx V(s; \\theta) $$
            $$ Q(s,a) \\approx Q(s,a; \\theta) $$
            
            ### Resources for Further Learning
            
            - [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Sutton & Barto
            - [Deep Reinforcement Learning](https://arxiv.org/abs/1810.06339) by Francois-Lavet et al.
            - [OpenAI Spinning Up](https://spinningup.openai.com/)
            """)
    
    # About Mode
    else:
        st.markdown('<p class="main-header">About This Project</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Reinforcement Learning for Optimization
        
        This interactive demo showcases the application of reinforcement learning techniques to solve complex 
        optimization problems in resource allocation. 
        
        ### Key Features
        
        - Implementation of multiple RL algorithms (PPO, DQN, A2C)
        - Interactive visualization of allocations and comparisons
        - Customizable problem parameters and constraints
        - Performance comparison with theoretical optimal solutions
        
        ### Technical Details
        
        The demo uses:
        - **TensorFlow** and **Stable Baselines3** for RL implementation
        - **Gymnasium** for the environment interface
        - **Streamlit** for the interactive web interface
        - **Matplotlib** and **Altair** for visualization
        
        ### About the Author
        
        This project was developed by [Steven Elliott Jr.](https://www.linkedin.com/in/steven-elliott-jr), a data scientist and machine learning engineer specializing in reinforcement learning and optimization.
        
        ### Repository
        
        The code for this project is available on [GitHub](https://github.com/stevenelliottjr/rl-optimization).
        
        ### License
        
        This project is licensed under the MIT License.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Applications
            
            Reinforcement learning for optimization can be applied to many real-world problems:
            
            - **Project Resource Allocation**
            - **Computing Resource Management**
            - **Supply Chain Optimization**
            - **Portfolio Management**
            - **Traffic Flow Optimization**
            - **Energy Grid Management**
            """)
        
        with col2:
            st.markdown("""
            ### References
            
            1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
            
            2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
            
            3. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
            
            4. Kool, W., van Hoof, H., & Welling, M. (2018). Attention, learn to solve routing problems! arXiv preprint arXiv:1803.08475.
            """)

if __name__ == "__main__":
    main()
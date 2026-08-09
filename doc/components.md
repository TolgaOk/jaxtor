# Component architecture

The sampled runtime path starts with the same environment protocol. The numbered
layers show its composition, while the lower branch shows exact tabular operations.
Each runtime node also shows the explicit state threaded through its calls.

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui, sans-serif","fontSize":"15px","lineColor":"#718096","edgeLabelBackground":"#ffffff"},"flowchart":{"curve":"basis","nodeSpacing":28,"rankSpacing":42,"padding":12}}}%%
flowchart TB
    subgraph environments["1 · ENVIRONMENTS | interchangeable backends"]
        direction LR
        tabular["TabularEnv<br/>jaxdp MDP"]
        gymnax["GymnaxEnv<br/>pure JAX"]
        mjx["MjxEnv<br/>MJX · XLA / Warp"]
        gym["GymEnv<br/>Gymnasium · EnvPool<br/>State: runtime handle · observations"]
    end

    env_api(["Env protocol<br/>init · reset · step · obs<br/>state: backend-specific"])
    tabular --> env_api
    gymnax --> env_api
    mjx --> env_api
    gym --> env_api

    subgraph sampling["2 · SAMPLING | episode lifecycle"]
        direction LR
        mc["Mc<br/>reset · time limit · episode queues<br/>state: key · env · observation · statistics"]
        vecmc["VecMc<br/>parallel vmap"]
        mc_api(["MC protocol<br/>sample(action, state)"])
        mc -->|scalar| mc_api
        mc -->|optional batching| vecmc
        vecmc -->|batched| mc_api
    end

    env_api --> mc

    subgraph interaction["3 · POLICY INTERACTION | closed sampling"]
        direction LR
        agent_api(["Agent protocol<br/>act(observation, state)"])
        imc["Imc<br/>action + transition<br/>state: mc · agent"]
        muimc["MuImc<br/>action + transition + log μ<br/>state: mc · agent"]
        imc_api(["IMC protocol<br/>sample(state)"])
        agent_api --> imc
        agent_api --> muimc
        mc_api --> imc
        mc_api --> muimc
        imc --> imc_api
        muimc --> imc_api
    end

    subgraph collection["4 · COLLECT / EVALUATE"]
        direction LR
        roll["Roll<br/>lax.scan over N steps"]
        mc_eval["McEval<br/>sampled episode metrics"]
    end

    imc_api --> roll
    imc_api --> mc_eval

    subgraph tabular_branch["TABULAR OPERATIONS"]
        direction LR
        mdp[["MDP arrays<br/>transition · reward · initial · terminal"]]
        sweep["Sweep<br/>stochastic all-(s,a) samples"]
        exp_sweep["ExpSweep<br/>exact forward / backward propagation"]
        tabular_eval["TabularEval<br/>convergence diagnostics"]
        value_api(["Tabular agent protocol<br/>q_vals(state, observations)"])
        mdp --> sweep
        mdp --> exp_sweep
        mdp --> tabular_eval
        value_api --> tabular_eval
    end

    mc_eval ~~~ mdp
    roll ~~~ value_api
    tabular -. exposes .-> mdp
    mc --> sweep

    subgraph consumers["CONSUMERS"]
        direction LR
        update["Learning update"]
        report["Evaluation · logging"]
    end

    roll -->|transition trajectory| update
    sweep -->|batched transitions| update
    exp_sweep -->|value / occupancy sequences| update
    mc_eval -->|episode metrics| report
    tabular_eval -->|convergence metrics| report

    classDef adapter fill:#E8F0FE,stroke:#4E73A8,color:#172A46,stroke-width:1.4px;
    classDef protocol fill:#FFFFFF,stroke:#52657A,color:#223247,stroke-width:2px;
    classDef sampler fill:#E4F4F1,stroke:#2B7A78,color:#153C3B,stroke-width:1.4px;
    classDef interactionNode fill:#F0EAF8,stroke:#76559A,color:#35254C,stroke-width:1.4px;
    classDef analysis fill:#FFF2DE,stroke:#B7791F,color:#51330B,stroke-width:1.4px;
    classDef data fill:#F2F4F7,stroke:#7C8798,color:#273444,stroke-width:1.2px;
    classDef consumer fill:#FFFFFF,stroke:#7C8798,color:#273444,stroke-width:1.2px,stroke-dasharray:5 3;

    class tabular,gymnax,mjx,gym adapter;
    class env_api,mc_api,agent_api,imc_api,value_api protocol;
    class mc,vecmc sampler;
    class imc,muimc,roll interactionNode;
    class mc_eval,sweep,exp_sweep,tabular_eval analysis;
    class mdp data;
    class update,report consumer;

    style environments fill:#F8FAFD,stroke:#C8D5E6,stroke-width:1px,color:#425466
    style sampling fill:#F7FCFA,stroke:#BBDAD4,stroke-width:1px,color:#425466
    style interaction fill:#FBF9FD,stroke:#D4C8E5,stroke-width:1px,color:#425466
    style collection fill:#FFFCF7,stroke:#E8D3AD,stroke-width:1px,color:#425466
    style tabular_branch fill:#FFFCF7,stroke:#E8D3AD,stroke-width:1px,color:#425466
    style consumers fill:#FAFBFC,stroke:#D4DAE2,stroke-width:1px,color:#425466
    linkStyle default stroke:#718096,stroke-width:1.35px;
```

Solid arrows show runtime composition and data flow. Dotted arrows show structural
relationships. Components are configured objects, while dynamic data lives in
explicit state pytrees. The composition can be wrapped in `jit`; `VecMc` uses
`vmap`, `Roll` and `McEval` use `scan`, and `grad` remains available along pure-JAX
environment paths.

## Host-backed Gymnasium boundary

`GymEnv` preserves the component and `State` split even though a mutable
Gymnasium environment cannot be a JAX pytree. The configured component contains
the runtime factory, array schema, and batching rules. `init(key)` creates a
scalar environment when `num_envs` is omitted, or a vector pool when it is an
integer, and returns the state that identifies the process-local runtime.

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui, sans-serif","fontSize":"15px","lineColor":"#718096","edgeLabelBackground":"#ffffff"},"flowchart":{"curve":"basis","nodeSpacing":30,"rankSpacing":44,"padding":12}}}%%
flowchart LR
    component["GymEnv<br/>factory · schema · batching rules"]
    init["init(key)<br/>create · seed · reset"]
    state[["GymEnv.State<br/>runtime [id · token]<br/>obs · reset_obs"]]
    mc["Mc.sample<br/>scalar environment logic"]
    batch["VecMc + custom_vmap<br/>one vector operation"]
    callback["io_callback<br/>host boundary"]
    store[("runtime store<br/>runtime_id → environment")]
    runtime["Gymnasium runtime<br/>Env when None · VectorEnv when N"]

    component --> init
    init -->|returns| state
    init -. allocates .-> store
    state -->|scalar fields| mc
    component -->|step / reset| mc
    mc -->|scalar| callback
    mc -->|vectorized| batch
    batch -->|one pool step| callback
    callback -->|runtime handle · action| store
    store --> runtime
    runtime -->|transition| callback
    callback -->|arrays · next token| state

    classDef adapter fill:#E8F0FE,stroke:#4E73A8,color:#172A46,stroke-width:1.4px;
    classDef stateNode fill:#F2F4F7,stroke:#7C8798,color:#273444,stroke-width:1.4px;
    classDef sampler fill:#E4F4F1,stroke:#2B7A78,color:#153C3B,stroke-width:1.4px;
    classDef boundary fill:#F0EAF8,stroke:#76559A,color:#35254C,stroke-width:1.4px;
    classDef external fill:#FFF2DE,stroke:#B7791F,color:#51330B,stroke-width:1.4px;

    class component,init adapter;
    class state stateNode;
    class mc,batch sampler;
    class callback boundary;
    class store,runtime external;

    linkStyle default stroke:#718096,stroke-width:1.35px;
```

The two-word runtime handle contains the process-local pool identity and the
sequencing token. The position along the `vmap` axis identifies a pool slot, so
no duplicate slot value is carried through every step. The callback returns the
next token, which becomes an input to the following callback. This data
dependency orders external steps inside `jit` and `scan`. State copies refer to
the same runtime, so each state should be consumed once. The runtime is released
with `env.close(state)`.

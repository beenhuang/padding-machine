//REPLACE-relay-padding-machine-REPLACE

/**
 * RBB the RELAY-side circuit padding machine on the general circuit.
 */  

  /* [ALLOCATE] : allocate memory for a relay machine */
  circpad_machine_spec_t *relay_machine = tor_malloc_zero(sizeof(circpad_machine_spec_t));

  /* [NAME] : relay machine name */
  relay_machine->name = "rbb_relay";

  /* [SIDE] : This is a relay-side machine */
  relay_machine->is_origin_side = 0;

  /* [HOP] : target hop, #1 is guard, #2 middle, and #3 exit. */
  //relay_machine->target_hopnum = 2;

  //relay_machine->conditions.min_hops = 2;
  //relay_machine->conditions.apply_state_mask = CIRCPAD_CIRC_STREAMS;
  //relay_machine->conditions.apply_purpose_mask = circpad_circ_purpose_to_mask(CIRCUIT_PURPOSE_C_GENERAL)|circpad_circ_purpose_to_mask(CIRCUIT_PURPOSE_C_CIRCUIT_PADDING);

  /* [ABSOLUTE LIMIT] : We can only allow up to 64k of padding cells */
  //relay_machine->allowed_padding_count = 65535;
  relay_machine->allowed_padding_count = 1500;

  /* [RELATIVE LIMIT]: padding/total, we can use [0~100]% here */
  relay_machine->max_padding_percent = 50;  

/********************  [START OF STATES]  *********************/

  // Random burst padding needs three states.
  // +One is the idle state
  // +Two is the state which decides when to add a fake burst
  // +Three is the state where the length of the fake burst is determined
  circpad_machine_states_init(relay_machine, 3);

  // The start state transitions to the second state when packets are sent(REB) or received(RBB)
  relay_machine->states[0].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  // Transition back to start when the infinity bin is sampled in either states two or three.
  // +In state two, the infinity bin signals that a fake burst should NOT be sent
  relay_machine->states[1].next_state[CIRCPAD_EVENT_INFINITY] = 0;
  // +In state three, the infinity bin signals that a fake burst should end
  relay_machine->states[2].next_state[CIRCPAD_EVENT_INFINITY] = 0;
  // Stop padding if the maximum padding per burst is reached.
  relay_machine->states[2].next_state[CIRCPAD_EVENT_LENGTH_COUNT] = 0;

  // Machine should transition to state three when padding is sent (ie. fake burst is active)
  relay_machine->states[1].next_state[CIRCPAD_EVENT_PADDING_SENT] = 2;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_SENT] = 2;

  // State Two Histogram definition:
  // +Two bins
  // +Infinity bin indicates not to start fake burst
  // +Zero bin immediantly starts fake burst
  relay_machine->states[1].histogram_len = 2;
  relay_machine->states[1].histogram_edges[0] = 0;
  relay_machine->states[1].histogram_edges[1] = 1;
  // 10% to start padding
  relay_machine->states[1].histogram[0] = 1;
  relay_machine->states[1].histogram[1] = 9;
  relay_machine->states[1].histogram_total_tokens = 10;
  relay_machine->states[1].use_rtt_estimate = 0;
  relay_machine->states[1].token_removal = CIRCPAD_TOKEN_REMOVAL_NONE;

  // State Three Histogram definition:
  // +Two bins
  // +Infinity bin indicates not end fake burst
  // +Zero bin continues burst with additional cell
  relay_machine->states[2].histogram_len = 2;
  relay_machine->states[2].histogram_edges[0] = 0;
  relay_machine->states[2].histogram_edges[1] = 1;
  // 50% to continue padding, 50% to end
  relay_machine->states[2].histogram[0] = 5;
  relay_machine->states[2].histogram[1] = 5;
  relay_machine->states[2].histogram_total_tokens = 10;
  relay_machine->states[2].use_rtt_estimate = 0;
  relay_machine->states[2].token_removal = CIRCPAD_TOKEN_REMOVAL_NONE;
  // Set maximum number of cells per fake burst.
  relay_machine->states[2].length_dist.type = CIRCPAD_DIST_UNIFORM;
  relay_machine->states[2].start_length = 5;
  relay_machine->states[2].max_length = 5;
  relay_machine->states[2].length_includes_nonpadding = 0;
  // Note: An alternative way of implementing this state would to solely end 
  // fake bursts based on the sampled histogram length condition. For example,
  // a uniform distribution from [0,6] could be used with a 100% chance to 
  // sample 0 usec delays to achieve equal likely hood of small and large 
  // fake bursts.
  // TODO: simulated experiments needed for better tuning

/*****************************  [END OF STATES]  *****************************/

  /* [MACHINE NUMBER] : get the machine number from global padding machine list */
  relay_machine->machine_num = smartlist_len(relay_padding_machines);

  /* [REGISTER] : the padding machine to the global list */
  circpad_register_padding_machine(relay_machine, relay_padding_machines);

  /* [LOGGING] : */
  log_info(LD_CIRC, "[RBB_RELAY] Registered the RBB relay padding machine (%u)", relay_machine->machine_num);


//REPLACE-relay-padding-machine-REPLACE

/**
 * august-mr: the RELAY-side circuit padding machine on the general circuit.
 */  

  /* [ALLOCATE] : allocate memory for a relay machine */
  circpad_machine_spec_t *relay_machine = tor_malloc_zero(sizeof(circpad_machine_spec_t));

  /* [NAME] : relay machine name */
  relay_machine->name = "august_relay";

  /* [SIDE] : This is a relay-side machine */
  relay_machine->is_origin_side = 0;

  /* [HOP] : target hop, #1 is guard, #2 middle, and #3 exit. */
  //relay_machine->target_hopnum = 2;

  /**
   * this machine's conditions
   * add the machine on the cricuit, when match all conditions, 
   */

  /* [APPLY/KEEP CIRCUIT_STATE] :  */
  //relay_machine->conditions.apply_state_mask = CIRCPAD_CIRC_OPENED;


  /* [ABSOLUTE LIMIT] : We can only allow up to 64k of padding cells */
  relay_machine->allowed_padding_count = 1500;

  /* [RELATIVE LIMIT]: padding/total, we can use [0~100]% here */
  relay_machine->max_padding_percent = 50;  

/********************  [START OF STATES]  *********************/

  /**
   * iat = MIN(iat_dist, dist_max_sample_usec)+dist_added_shift_usec
   * length = MIN((start_length + length_dist), max_length)
   */  

  /* machine has following states:
   *  0. [START]: start state.
   *  1. [WAIT]: do nothing
   */  

  /**
   * CIRCPAD_DIST_UNIFORM
   * CIRCPAD_DIST_LOGISTIC
   * CIRCPAD_DIST_LOG_LOGISTIC
   * CIRCPAD_DIST_GEOMETRIC
   * CIRCPAD_DIST_WEIBULL
   * CIRCPAD_DIST_PARETO
   */ 
  
  /* [STATE INIT] : */
  circpad_machine_states_init(relay_machine, 2);

  /* 0. [START] state: transite to the WAIT state when relay received a NEGOTIATION cell. */
  relay_machine->states[0].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* 1. [WAIT] state: do nothing. */
  relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;
  relay_machine->states[1].next_state[CIRCPAD_EVENT_PADDING_RECV] = 1;
  relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;


/*****************************  [END OF STATES]  *****************************/

  /* [MACHINE NUMBER] : get the machine number from global padding machine list */
  relay_machine->machine_num = smartlist_len(relay_padding_machines);

  /* [REGISTER] : the padding machine to the global list */
  circpad_register_padding_machine(relay_machine, relay_padding_machines);

  /* [LOGGING] : */
  log_info(LD_CIRC, "[AUGUST_RELAY] Registered the AUGUST relay padding machine (%u)", relay_machine->machine_num);



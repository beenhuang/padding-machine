//REPLACE-relay-padding-machine-REPLACE

/**
 * october the RELAY-side circuit padding machine on the general circuit.
 */  

  /* [ALLOCATE] : allocate memory for a relay machine */
  circpad_machine_spec_t *relay_machine = tor_malloc_zero(sizeof(circpad_machine_spec_t));

  /* [NAME] : relay machine name */
  relay_machine->name = "october_relay";

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
   *  1. [WAIT]: transite to extend, fake or break.
   *  2. [BREAK]: break burst.
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
  circpad_machine_states_init(relay_machine, 3);

  /* 0. [START] state: transite to the WAIT state when relay received a NEGOTIATION cell. */
  relay_machine->states[0].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* 1. [WAIT] state: transite to EXTEND or FAKE states. */
  //relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;
  //relay_machine->states[1].next_state[CIRCPAD_EVENT_PADDING_RECV] = 1;
  relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 2;

  /* 2. [BREAKBURST] state: */
  // 2.1 [RECV]: max cells received continuously
  relay_machine->states[2].contin_recv_length_dist.type = CIRCPAD_DIST_WEIBULL;
  relay_machine->states[2].contin_recv_length_dist.param1 = 1.1939537311219628;
  relay_machine->states[2].contin_recv_length_dist.param2 = 2.2388583813756533;
  //relay_machine->states[2].contin_recv_start_length = 1;
  //relay_machine->states[2].contin_recv_max_length = 30;

  // 2.2 [SEND]: max padding cells sent continuously 
  relay_machine->states[2].contin_padding_sent_length_dist.type =  CIRCPAD_DIST_PARETO;
  relay_machine->states[2].contin_padding_sent_length_dist.param1 = 7.009539453953314;
  relay_machine->states[2].contin_padding_sent_length_dist.param2 = -1.7523848634883286;
  //relay_machine->states[2].contin_padding_sent_start_length = 2;
  //relay_machine->states[2].contin_padding_sent_max_length = 5;

  // includes padding cell received
  relay_machine->states[2].contin_includes_padding_recv = 1;

  // 2.3 [BREAKBURST] state transition.
  relay_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_SENT] = 2;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 2;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_RECV] = 2;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;  


/*****************************  [END OF STATES]  *****************************/

  /* [MACHINE NUMBER] : get the machine number from global padding machine list */
  relay_machine->machine_num = smartlist_len(relay_padding_machines);

  /* [REGISTER] : the padding machine to the global list */
  circpad_register_padding_machine(relay_machine, relay_padding_machines);

  /* [LOGGING] : */
  log_info(LD_CIRC, "[OCTOBER_RELAY] Registered the OCTOBER relay padding machine (%u)", relay_machine->machine_num);



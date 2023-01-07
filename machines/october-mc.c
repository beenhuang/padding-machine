//REPLACE-client-padding-machine-REPLACE

/**
 * october CLIENT-side circuit padding machine on the general circuit.
 */  

  /* [ALLOCATE] : allocate memory for a client machine */
  circpad_machine_spec_t *client_machine = tor_malloc_zero(sizeof(circpad_machine_spec_t));

  /* [NAME] : client machine name */
  client_machine->name = "october_client";

  /* [SIDE] : This is a client-side machine */
  client_machine->is_origin_side = 1;
  
  /* [HOP] : target hop, #1 is guard, #2 middle, and #3 exit. */
  client_machine->target_hopnum = 2;

  /* [LIFETIME] : do not close circuit */ 
  //client_machine->manage_circ_lifetime = 1;


  /**
   * this machine's conditions
   * add the machine on the cricuit, when match all conditions, 
   */

  /* [REDUCED] : concensus parameter */
  // client_machine->conditions.reduced_padding_ok = 1;

  /* [APPLY/KEEP CIRCUIT_STATE] :  */
  client_machine->conditions.apply_state_mask = CIRCPAD_CIRC_STREAMS;
  // client_machine->conditions.keep_state_mask = CIRCPAD_CIRC_STREAMS;

  /* [APPLY/KEEP CIRCUIT_PURPOSE] : */
  client_machine->conditions.apply_purpose_mask = CIRCPAD_PURPOSE_ALL;
  // client_machine->conditions.keep_purpose_mask = CIRCPAD_PURPOSE_ALL;

  

  /* [ABSOLUTE LIMIT] : We can only allow up to 64k of padding cells */
  client_machine->allowed_padding_count = 1500;

  /* [RELATIVE LIMIT]: padding/total, we can use [0~100]% here */
  client_machine->max_padding_percent = 50;

/******************  [START OF STATES]  *****************/

  /**
   * iat = MIN(iat_dist, dist_max_sample_usec)+dist_added_shift_usec
   * length = MIN((start_length + length_dist), max_length)
   */  

  /* machine has following states:
   *  0. [START]: start state.
   *  1. [WAIT]: transite to extend, fake or break.
   *  2. [BREAK]: break burst, break server-response burst.
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
  circpad_machine_states_init(client_machine, 3);

  /* 0. [START] state: transite to the wait state when client send a NEGOTIATION cell. */
  client_machine->states[0].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;

  /* 1. [WAIT] state: transite to EXTEND/FAKE/BREAK states. */
  //client_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;
  //client_machine->states[1].next_state[CIRCPAD_EVENT_PADDING_RECV] = 2;
  client_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 2;

  /* 2. [BREAKBURST] state: */
  // 2.1 [RECV]: max cells received continuously
  client_machine->states[2].contin_recv_length_dist.type = CIRCPAD_DIST_WEIBULL;
  client_machine->states[2].contin_recv_length_dist.param1 = 1.3832926042292748;
  client_machine->states[2].contin_recv_length_dist.param2 = 8.766888541863576;
  //client_machine->states[2].contin_recv_start_length = 1;
  //client_machine->states[2].contin_recv_max_length = 30;

  // 2.2 [SEND]: max padding cells sent continuously 
  client_machine->states[2].contin_padding_sent_length_dist.type =  CIRCPAD_DIST_PARETO;
  client_machine->states[2].contin_padding_sent_length_dist.param1 = 1.9667283364576538;
  client_machine->states[2].contin_padding_sent_length_dist.param2 = 0.05282296143414936;
  //client_machine->states[2].contin_padding_sent_start_length = 2;
  //client_machine->states[2].contin_padding_sent_max_length = 5;

  // includes padding cell received
  client_machine->states[2].contin_includes_padding_recv = 1;

  // 2.3 [BREAKBURST] state transition.
  client_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_SENT] = 2;
  client_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 2;
  client_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_RECV] = 2;
  client_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;

/*****************************  [END OF STATES]  *****************************/

  /* [MACHINE NUMBER] : get the machine number from global padding machine list */
  client_machine->machine_num = smartlist_len(origin_padding_machines);

  /* [REGISTER] : the padding machine to the global list */
  circpad_register_padding_machine(client_machine, origin_padding_machines);

  /* [LOGGING] : */
  log_info(LD_CIRC, "[OCTOBER_CLIENT] Registered the OCTOBER client padding machine (%u)", client_machine->machine_num);
  
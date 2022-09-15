/**
 * july client-side circuit padding machine on the general circuit.
 */  

  circpad_machine_spec_t *client_machine = tor_malloc_zero(sizeof(circpad_machine_spec_t));

  /* client machine name */
  client_machine->name = "july_client";

  /* This is a client-side machine */
  client_machine->is_origin_side = 1;
  
  /* target hop: #1 is guard, #2 middle, #3 exit */
  client_machine->target_hopnum = 2;

  /** If this flag is enabled, don't close circuits that use this machine even
   *  if another part of Tor wants to close this circuit. 
   */ 
  //client_machine->manage_circ_lifetime = 1;

  /**
   *  absolute and relative overhead limits. 
   */

  // absolute limit: We can only allow up to 64k of padding cells
  //client_machine->allowed_padding_count = 1500;

  // relative limit: padding/total, we can use [0~100]% here.
  //client_machine->max_padding_percent = 50;


  /**
   * this machine's conditions.
   */

  // reduced padding machine: 
  // client_machine->conditions.reduced_padding_ok = 1;

  client_machine->conditions.apply_state_mask = CIRCPAD_CIRC_STREAMS;
  // client_machine->conditions.keep_state_mask = CIRCPAD_CIRC_STREAMS;

  // circuit_state_mask: 
  client_machine->conditions.apply_purpose_mask = CIRCPAD_PURPOSE_ALL;
  // client_machine->conditions.keep_purpose_mask = CIRCPAD_PURPOSE_ALL;


  /*************** start of states. **************/
  /*
   * length_dist : how many padding cells can we send in this state.
   * iat_dist: how long can we wait util sending the next padding cell.
   *
   * actual cell count = min((start_length + length_dist), max_length)
   * actual iat = min(iat_dist, dist_max_sample_usec) + dist_added_shift_usec
   */  

  /* machine has following states:
   *  0. start: start state.
   *  1. wait: either extend, fake and break.
   *  2. extend (extend burst): extend the length of the client-request.
   *  3. fake (fake client-request): inject a fake client-request.
   *  4. break (extend burst): break server-response burst.
   */

  circpad_machine_states_init(client_machine, 5);

  // [start] state: transite to the wait state when client send first client-request.
  client_machine->states[0].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;

  /* [wait] state: transite from wait state to extend/fake/break state. */
  // [wait] state transition:
  client_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 2;
  client_machine->states[1].next_state[CIRCPAD_EVENT_PADDING_RECV] = 3;
  client_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 4;

  // [extend] state: 
  client_machine->states[2].length_dist.type = CIRCPAD_DIST_UNIFORM;
  client_machine->states[2].length_dist.param1 = 0;
  client_machine->states[2].length_dist.param2 = 2;
  client_machine->states[2].start_length = 2;
  client_machine->states[2].max_length = 4;
  client_machine->states[2].iat_dist.type = CIRCPAD_DIST_PARETO;
  client_machine->states[2].iat_dist.param1 = 3.3;
  client_machine->states[2].iat_dist.param2 = 7.5;
  // client_machine->states[2].dist_added_shift_usec = 100;
  client_machine->states[2].dist_max_sample_usec = 9445;
  
  // extend state tansitions:
  client_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_SENT] = 2;
  client_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 2;
  client_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_RECV] = 1;
  client_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* [fake] state: */
  client_machine->states[3].length_dist.type = CIRCPAD_DIST_UNIFORM;
  client_machine->states[3].length_dist.param1 = 0;
  client_machine->states[3].length_dist.param2 = 4;
  client_machine->states[3].start_length = 2;
  client_machine->states[3].max_length = 6;
  client_machine->states[3].iat_dist.type = CIRCPAD_DIST_PARETO;
  client_machine->states[3].iat_dist.param1 = 3.3;
  client_machine->states[3].iat_dist.param2 = 7.1;
  // client_machine->states[3].dist_added_shift_usec = 100;
  client_machine->states[3].dist_max_sample_usec = 9445;
  
  // fake state transition.
  client_machine->states[3].next_state[CIRCPAD_EVENT_PADDING_SENT] = 3;
  client_machine->states[3].next_state[CIRCPAD_EVENT_PADDING_RECV] = 3;
  client_machine->states[3].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;
  client_machine->states[3].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* [BREAKBURST] state: */
  // received cell continuously
  //client_machine->states[4].length_dist.type = CIRCPAD_DIST_NONE;
  // count cell received continuously
  client_machine->states[4].contin_recv_length_dist.type = CIRCPAD_DIST_UNIFORM;
  client_machine->states[4].contin_recv_length_dist.param1 = 0;
  client_machine->states[4].contin_recv_length_dist.param2 = 10;
  client_machine->states[4].contin_recv_start_length = 20;
  client_machine->states[4].contin_recv_max_length = 30;
  // send padding continuously 
  client_machine->states[4].contin_padding_sent_length_dist.type = CIRCPAD_DIST_UNIFORM;
  client_machine->states[4].contin_padding_sent_length_dist.param1 = 0;
  client_machine->states[4].contin_padding_sent_length_dist.param2 = 4;
  client_machine->states[4].contin_padding_sent_start_length = 2;
  client_machine->states[4].contin_padding_sent_max_length = 6;

  // includes received padding cell
  client_machine->states[4].contin_includes_padding_recv = 1;


  // [BREAKBURST] state transition.
  client_machine->states[4].next_state[CIRCPAD_EVENT_PADDING_SENT] = 4;
  client_machine->states[4].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 4;
  client_machine->states[4].next_state[CIRCPAD_EVENT_PADDING_RECV] = 4;
  client_machine->states[4].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;


  /*****************************  end of states  *****************************/

  /* Register the machine */
  // set machine_num with the last one of the global list.
  client_machine->machine_num = smartlist_len(origin_padding_machines);

  // register padding machine to the global list.
  circpad_register_padding_machine(client_machine, origin_padding_machines);

  // logging in the information level.
  log_info(LD_CIRC, "[JULY_PADDING] Registered my own client padding machine (%u)", 
           client_machine->machine_num);
  
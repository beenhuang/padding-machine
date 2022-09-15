/**
 *
 * july relay-side circuit padding machine on the general circuit.
 *
 */  

  circpad_machine_spec_t *relay_machine = tor_malloc_zero(sizeof(circpad_machine_spec_t));

  /* relay machine name */
  relay_machine->name = "july_relay";

  /* This is a relay-side machine */
  relay_machine->is_origin_side = 0;

  /* target hop: #1 is guard, #2 middle, #3 exit */
  // relay_machine->target_hopnum = 2;

  /**
   *  absolute and relative overhead limits. 
   */
  // absolute limit: We can only allow up to 64k of padding cells
  relay_machine->allowed_padding_count = 1500;

  // relative limit: padding/total, we can use [0~100]% here.
  relay_machine->max_padding_percent = 50;


  /**
   * this machine's conditions.
   * If match all conditions, then add the machine on the cricuit.
   * 
   * relay-side do not set apply_purpose_mask.
   * relay-side can work in the CIRCPAD_CIRC_OPENED state.
   */
  
  // circuit_state_mask
  //relay_machine->conditions.apply_state_mask = CIRCPAD_CIRC_OPENED;

  /**********************  start of states  *********************/
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
   *  2. extend: extend the length of the client-request.
   *  3. fake: inject a fake client-request.
   *  4. break: break server-response burst.
   */
  
  circpad_machine_states_init(relay_machine, 4);

  // [start] state: transite to the wait state when client send first client-request.
  relay_machine->states[0].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;


  /* [wait] state: transite from wait state to extend/fake/break state. */
  // [wait] state transition:
  relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 2;
  relay_machine->states[1].next_state[CIRCPAD_EVENT_PADDING_RECV] = 3;
  relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;
  //relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 4;


  // [extend] state: 
  relay_machine->states[2].length_dist.type = CIRCPAD_DIST_UNIFORM;
  relay_machine->states[2].length_dist.param1 = 0;
  relay_machine->states[2].length_dist.param2 = 5;
  relay_machine->states[2].start_length = 5;
  relay_machine->states[2].max_length = 10;
  relay_machine->states[2].iat_dist.type = CIRCPAD_DIST_LOGISTIC;
  relay_machine->states[2].iat_dist.param1 = 5.2;
  relay_machine->states[2].iat_dist.param2 = 5.4;
  // relay_machine->states[2].dist_added_shift_usec = 100;
  relay_machine->states[2].dist_max_sample_usec = 94733;
  
  // [extend] state tansitions: 
  relay_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_SENT] = 2;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 2;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_RECV] = 1;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* [fake] state: */
  relay_machine->states[3].length_dist.type = CIRCPAD_DIST_UNIFORM;
  relay_machine->states[3].length_dist.param1 = 0;
  relay_machine->states[3].length_dist.param2 = 20;
  relay_machine->states[3].start_length = 20;
  relay_machine->states[3].max_length = 40;
  relay_machine->states[3].iat_dist.type = CIRCPAD_DIST_UNIFORM;
  relay_machine->states[3].iat_dist.param1 = 4.2;
  relay_machine->states[3].iat_dist.param2 = 7.9;
  // relay_machine->states[3].dist_added_shift_usec = 100;
  relay_machine->states[3].dist_max_sample_usec = 55878;
  
  // [fake] state transition:
  relay_machine->states[3].next_state[CIRCPAD_EVENT_PADDING_SENT] = 3;
  relay_machine->states[3].next_state[CIRCPAD_EVENT_PADDING_RECV] = 3;
  relay_machine->states[3].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;
  relay_machine->states[3].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* [BREAKBURST] state: 
  relay_machine->states[4].length_dist.type = CIRCPAD_DIST_LOGISTIC;
  relay_machine->states[4].length_dist.param1 = 1.6;
  relay_machine->states[4].length_dist.param2 = 6.1;
  relay_machine->states[4].start_length = 1;
  // relay_machine->states[4].max_length = 22;
  relay_machine->states[4].iat_dist.type = CIRCPAD_DIST_UNIFORM;
  relay_machine->states[4].iat_dist.param1 = 4.2;
  relay_machine->states[4].iat_dist.param2 = 7.9;
  // relay_machine->states[4].dist_added_shift_usec = 100;
  relay_machine->states[4].dist_max_sample_usec = 10000;
  
  // [BREAKBURST] state transition:
  relay_machine->states[4].next_state[CIRCPAD_EVENT_PADDING_SENT] = 4;
  relay_machine->states[4].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;
  */

  /*************************  end of states  **************************/


  /* Register the machine */
  // set machine_num with the last one of the global list.
  relay_machine->machine_num = smartlist_len(relay_padding_machines);

  // register padding machine to the global list.
  circpad_register_padding_machine(relay_machine, relay_padding_machines);

  // logging in the information level.
  log_info(LD_CIRC, "[JULY_PADDING] Registered my own relay padding machine (%u)", relay_machine->machine_num);


/**
 * august the relay-side circuit padding machine on the general circuit.
 */  

  /* [ALLOCATE] : allocate memory for a relay machine */
  circpad_machine_spec_t *relay_machine = tor_malloc_zero(sizeof(circpad_machine_spec_t));

  /* [NAME] : relay machine name */
  relay_machine->name = "august_relay";

  /* [SIDE] : This is a relay-side machine */
  relay_machine->is_origin_side = 0;

  /* [HOP] : target hop, #1 is guard, #2 middle, and #3 exit. */
  //relay_machine->target_hopnum = 2;

  /* [ABSOLUTE LIMIT] : We can only allow up to 64k of padding cells */
  relay_machine->allowed_padding_count = 1500;

  /* [RELATIVE LIMIT]: padding/total, we can use [0~100]% here */
  relay_machine->max_padding_percent = 50;


  /**
   * this machine's conditions
   * add the machine on the cricuit, when match all conditions, 
   */

  /* [APPLY/KEEP CIRCUIT_STATE] :  */
  //relay_machine->conditions.apply_state_mask = CIRCPAD_CIRC_OPENED;


  /********************  [START OF STATES]  *********************/
  
  /* short define for sampling uniformly random [0, 1.0]  */
  const struct uniform_t my_uniform = {
    .base = UNIFORM(my_uniform),
    .a = 0.0,
    .b = 1.0,
  };

  #define CIRCPAD_UNI_RAND (dist_sample(&my_uniform.base))

  /* uniformly random select a distribution parameters between [0, 10] */
  #define CIRCPAD_RAND_DIST_PARAM1 (CIRCPAD_UNI_RAND *10) 
  #define CIRCPAD_RAND_DIST_PARAM2 (CIRCPAD_UNI_RAND *10)

  /**
   * length = MIN((start_length + length_dist), max_length)
   * iat = MIN(iat_dist, dist_max_sample_usec)+dist_added_shift_usec
   */  

  /* machine has following states:
   *  0. [START]: start state.
   *  1. [WAIT]: transite to extend, fake or break.
   *  2. [EXTEND]: extend burst, extend the length of the client-request.
   *  3. [FAKE]: fake client-request, inject a fake client-request.
   */  
  
  /* [STATE INIT] : */
  circpad_machine_states_init(relay_machine, 4);

  /* 0. [START] state: transite to the WAIT state when relay received a NEGOTIATION cell. */
  relay_machine->states[0].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* 1. [WAIT] state: transite to EXTEND or FAKE states. */
  relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 2;
  relay_machine->states[1].next_state[CIRCPAD_EVENT_PADDING_RECV] = 3;
  relay_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* 2. [EXTEND] state: */ 
  relay_machine->states[2].length_dist.type = CIRCPAD_DIST_GEOMETRIC;
  relay_machine->states[2].length_dist.param1 = 0.648578;
  //relay_machine->states[2].length_dist.param2 = ;
  relay_machine->states[2].start_length = 2;
  //relay_machine->states[2].max_length = 10;

  relay_machine->states[2].iat_dist.type = CIRCPAD_DIST_LOG_LOGISTIC;
  relay_machine->states[2].iat_dist.param1 = 26.22286;
  relay_machine->states[2].iat_dist.param2 = 21.0;
  relay_machine->states[2].dist_added_shift_usec = 15;
  relay_machine->states[2].dist_max_sample_usec = 13000;
  
  // EXTEND state tansitions: 
  relay_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_SENT] = 2;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 2;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_RECV] = 1;
  relay_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* 3. [FAKE] state: */
  relay_machine->states[3].length_dist.type = CIRCPAD_DIST_LOG_LOGISTIC;
  relay_machine->states[3].length_dist.param1 = 9.83312;
  relay_machine->states[3].length_dist.param2 = 0.86490;
  //relay_machine->states[3].start_length = 4;
  relay_machine->states[3].max_length = 20;

  relay_machine->states[3].iat_dist.type = CIRCPAD_DIST_LOGISTIC;
  relay_machine->states[3].iat_dist.param1 = 289.31193;
  relay_machine->states[3].iat_dist.param2 = 1709.90169;
  relay_machine->states[3].dist_added_shift_usec = 15;
  relay_machine->states[3].dist_max_sample_usec = 20000;
  
  // [fake] state transition:
  relay_machine->states[3].next_state[CIRCPAD_EVENT_PADDING_SENT] = 3;
  relay_machine->states[3].next_state[CIRCPAD_EVENT_PADDING_RECV] = 3;
  relay_machine->states[3].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;
  relay_machine->states[3].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /*****************************  [END OF STATES]  *****************************/


  /* [MACHINE NUMBER] : get the machine number from global padding machine list */
  relay_machine->machine_num = smartlist_len(relay_padding_machines);

  /* [REGISTER] : the padding machine to the global list */
  circpad_register_padding_machine(relay_machine, relay_padding_machines);

  /* [LOGGING] : */
  log_info(LD_CIRC, "[AUGUST_RELAY] Registered the AUGUST relay padding machine (%u)", relay_machine->machine_num);


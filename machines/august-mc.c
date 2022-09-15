/**
 * august client-side circuit padding machine on the general circuit.
 */  

  /* [ALLOCATE] : allocate memory for a client machine */
  circpad_machine_spec_t *client_machine = tor_malloc_zero(sizeof(circpad_machine_spec_t));

  /* [NAME] : client machine name */
  client_machine->name = "august_client";

  /* [SIDE] : This is a client-side machine */
  client_machine->is_origin_side = 1;
  
  /* [HOP] : target hop, #1 is guard, #2 middle, and #3 exit. */
  client_machine->target_hopnum = 2;

  /* [LIFETIME] : do not close circuit */ 
  //client_machine->manage_circ_lifetime = 1;

  /* [ABSOLUTE LIMIT] : We can only allow up to 64k of padding cells */
  client_machine->allowed_padding_count = 1500;

  /* [RELATIVE LIMIT]: padding/total, we can use [0~100]% here */
  client_machine->max_padding_percent = 50;

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


  /******************  [START OF STATES]  *****************/

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
   *  4. [BREAK]: break burst, break server-response burst.
   */

  /* [STATE INIT] : */
  circpad_machine_states_init(client_machine, 5);

  /* 0. [START] state: transite to the wait state when client send a NEGOTIATION cell. */
  client_machine->states[0].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;

  /* 1. [WAIT] state: transite to EXTEND/FAKE/BREAK states. */
  client_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 2;
  client_machine->states[1].next_state[CIRCPAD_EVENT_PADDING_RECV] = 3;
  client_machine->states[1].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 4;

  /* 2. [EXTEND] state: */
  client_machine->states[2].length_dist.type = CIRCPAD_DIST_GEOMETRIC;
  client_machine->states[2].length_dist.param1 = 0.90;
  //client_machine->states[2].length_dist.param2 = CIRCPAD_RAND_DIST_PARAM2;
  //client_machine->states[2].start_length = 1;
  client_machine->states[2].max_length = 2;

  client_machine->states[2].iat_dist.type = CIRCPAD_DIST_LOG_LOGISTIC;
  client_machine->states[2].iat_dist.param1 = 348.84538;
  client_machine->states[2].iat_dist.param2 = 21.0;
  client_machine->states[2].dist_added_shift_usec = 20;
  client_machine->states[2].dist_max_sample_usec = 13000;
  
  // EXTEND state tansitions:
  client_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_SENT] = 2;
  client_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 2;
  client_machine->states[2].next_state[CIRCPAD_EVENT_PADDING_RECV] = 1;
  client_machine->states[2].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* 3. [FAKE] state: */
  client_machine->states[3].length_dist.type = CIRCPAD_DIST_GEOMETRIC;
  client_machine->states[3].length_dist.param1 = 0.648578;
  //client_machine->states[3].length_dist.param2 = CIRCPAD_RAND_DIST_PARAM2;
  //client_machine->states[3].start_length = 4;
  client_machine->states[3].max_length = 6;

  client_machine->states[3].iat_dist.type = CIRCPAD_DIST_LOG_LOGISTIC;
  client_machine->states[3].iat_dist.param1 = 348.84538;
  client_machine->states[3].iat_dist.param2 = 21.0;
  client_machine->states[3].dist_added_shift_usec = 20;
  client_machine->states[3].dist_max_sample_usec = 13000;
  
  // FAKE state transition.
  client_machine->states[3].next_state[CIRCPAD_EVENT_PADDING_SENT] = 3;
  client_machine->states[3].next_state[CIRCPAD_EVENT_PADDING_RECV] = 3;
  client_machine->states[3].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;
  client_machine->states[3].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 1;

  /* 4. [BREAKBURST] state: */
  // max cells received continuously
  client_machine->states[4].contin_recv_length_dist.type = CIRCPAD_DIST_LOG_LOGISTIC;
  client_machine->states[4].contin_recv_length_dist.param1 = 9.83312;
  client_machine->states[4].contin_recv_length_dist.param2 = 1.13234;
  client_machine->states[4].contin_recv_start_length = 1;
  //client_machine->states[4].contin_recv_max_length = 30;

  // max padding cells sent continuously 
  client_machine->states[4].contin_padding_sent_length_dist.type = CIRCPAD_DIST_GEOMETRIC;
  client_machine->states[4].contin_padding_sent_length_dist.param1 = 0.648578;
  //client_machine->states[4].contin_padding_sent_length_dist.param2 = 4;
  //client_machine->states[4].contin_padding_sent_start_length = 2;
  client_machine->states[4].contin_padding_sent_max_length = 5;

  // includes padding cell received
  client_machine->states[4].contin_includes_padding_recv = 1;

  // [BREAKBURST] state transition.
  client_machine->states[4].next_state[CIRCPAD_EVENT_PADDING_SENT] = 4;
  client_machine->states[4].next_state[CIRCPAD_EVENT_NONPADDING_RECV] = 4;
  client_machine->states[4].next_state[CIRCPAD_EVENT_PADDING_RECV] = 4;
  client_machine->states[4].next_state[CIRCPAD_EVENT_NONPADDING_SENT] = 1;

  /*****************************  [END OF STATES]  *****************************/

  /* [MACHINE NUMBER] : get the machine number from global padding machine list */
  client_machine->machine_num = smartlist_len(origin_padding_machines);

  /* [REGISTER] : the padding machine to the global list */
  circpad_register_padding_machine(client_machine, origin_padding_machines);

  /* [LOGGING] : */
  log_info(LD_CIRC, "[AUGUST_CLIENT] Registered the AUGUST client padding machine (%u)", client_machine->machine_num);
  